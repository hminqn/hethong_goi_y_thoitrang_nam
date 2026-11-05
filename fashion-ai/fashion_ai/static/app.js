const $ = selector => document.querySelector(selector);
const token = $('meta[name="csrf-token"]').content;
const canvas = $('#preview');
const context = canvas.getContext('2d');
let bitmap = null;
let region = null;
let start = null;
let busy = false;

function message(text) { $('#message').textContent = text; }
function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('button, input, select').forEach(node => { node.disabled = value; });
}
async function post(url, body, json = false) {
  const headers = { 'X-CSRF-Token': token };
  if (json) headers['Content-Type'] = 'application/json';
  const response = await fetch(url, { method: 'POST', headers, body: json ? JSON.stringify(body) : body });
  const data = await response.json().catch(() => ({ error: 'Máy chủ chưa trả được kết quả. Kiểm tra cửa sổ đang chạy ứng dụng.' }));
  if (!response.ok) throw new Error(data.error || 'Không xử lý được yêu cầu.');
  return data;
}
function draw() {
  if (!bitmap) return;
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  if (region) {
    context.strokeStyle = '#c77b38';
    context.lineWidth = 3;
    context.strokeRect(region.x, region.y, region.w, region.h);
  }
}
function point(event) {
  const bounds = canvas.getBoundingClientRect();
  return { x: Math.max(0, Math.min(canvas.width, (event.clientX - bounds.left) * canvas.width / bounds.width)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - bounds.top) * canvas.height / bounds.height)) };
}
canvas.onpointerdown = event => {
  if (!bitmap || busy || $('#mode').value !== 'manual') return;
  start = point(event);
  canvas.setPointerCapture(event.pointerId);
};
canvas.onpointermove = event => {
  if (!start) return;
  const current = point(event);
  region = { x: Math.min(current.x, start.x), y: Math.min(current.y, start.y), w: Math.abs(current.x - start.x), h: Math.abs(current.y - start.y) };
  draw();
};
canvas.onpointerup = () => { start = null; };
canvas.onpointercancel = () => { start = null; };
$('#mode').onchange = () => { $('#manual-fields').hidden = $('#mode').value !== 'manual'; };
$('#photo').onchange = async () => {
  const file = $('#photo').files[0];
  if (!file) return;
  if (file.size > 12 * 1024 * 1024 || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    $('#photo').value = '';
    message('Chọn ảnh JPG, PNG hoặc WebP tối đa 12 MB.');
    return;
  }
  try {
    const next = await createImageBitmap(file, { imageOrientation: 'from-image' });
    if (next.width * next.height > 25000000) { next.close(); throw new Error('Ảnh vượt giới hạn 25 triệu pixel.'); }
    bitmap?.close();
    bitmap = next;
    const scale = Math.min(1, 1000 / Math.max(bitmap.width, bitmap.height));
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.hidden = false;
    $('#placeholder').hidden = true;
    region = null;
    $('#detections').replaceChildren();
    $('#results').hidden = true;
    draw();
    message('Đã chọn ảnh. Chọn cách tách món rồi nhấn tìm món.');
  } catch (error) { message(error.message || 'Không đọc được ảnh.'); }
};
$('#upload-form').onsubmit = async event => {
  event.preventDefault();
  if (busy) return;
  if ($('#mode').value === 'manual' && (!region || region.w < 10 || region.h < 10)) {
    message('Hãy kéo khoanh vùng món muốn giữ trên ảnh.');
    return;
  }
  const form = new FormData();
  form.append('image', $('#photo').files[0]);
  form.append('mode', $('#mode').value);
  form.append('role', $('#role').value);
  if (region) for (const [key, value] of Object.entries(region)) form.append(key, value / (['x', 'w'].includes(key) ? canvas.width : canvas.height));
  setBusy(true);
  message('Đang xử lý ảnh trên máy…');
  try {
    const result = await post('/analyze', form);
    $('#detections').replaceChildren();
    result.detections.forEach(row => {
      const card = element('article'); card.className = 'crop';
      const image = element('img'); image.src = row.image; image.alt = row.category;
      const description = row.category + (row.score === null ? ' · tự khoanh vùng' : ' · điểm nhận diện ' + (row.score * 100).toFixed(1) + '%');
      const choose = element('button', 'Giữ món này ↗'); choose.type = 'button';
      choose.onclick = () => recommend({ index: row.index });
      card.append(image, element('p', description), choose);
      $('#detections').append(card);
    });
    message(result.message || 'Chọn món muốn giữ để xem các bộ phối.');
  } catch (error) { message(error.message); }
  finally { setBusy(false); }
};

async function recommend(selection) {
  if (busy) return;
  setBusy(true);
  message('Đang tìm ứng viên, chấm tương thích và chọn bằng MMR…');
  try {
    const result = await post('/recommend', selection, true);
    $('#outfits').replaceChildren();
    result.outfits.forEach((row, index) => {
      const card = element('article'); card.className = 'outfit';
      card.append(element('h3', 'Bộ ' + (index + 1)));
      const pieces = element('div'); pieces.className = 'outfit-items';
      const items = [{ name: result.kept.name + ' · món giữ lại', image: result.kept.image },
        ...row.items.map(item => ({ name: item.name, image: '/catalog-images/' + item.id }))];
      items.forEach(item => {
        const piece = element('div'); const image = element('img'); image.src = item.image; image.alt = item.name;
        piece.append(image, element('p', item.name)); pieces.append(piece);
      });
      const scores = element('p', 'Điểm tổng hợp ' + row.score.toFixed(3) + ' · RF ' + row.compatibility.toFixed(3) + ' · Luật ' + row.rule_score.toFixed(3) + ' · Cosine ' + row.similarity.toFixed(3));
      scores.className = 'scores';
      const details = element('details'); details.append(element('summary', 'Xem bằng chứng gợi ý'));
      details.append(element('p', 'Thuộc tính truy vấn được tham chiếu từ món gần ảnh nhất trong kho: ' + row.attribute_proxy));
      row.matched_rules.forEach(rule => details.append(element('p', rule.antecedent.join(' + ') + ' → ' + rule.consequent.join(' + ') + ' · confidence ' + rule.confidence.toFixed(2) + ' · lift ' + rule.lift.toFixed(2))));
      if (!row.matched_rules.length) details.append(element('p', 'Không có luật phù hợp vượt ngưỡng cho bộ này.'));
      card.append(pieces, scores, details); $('#outfits').append(card);
    });
    if (!result.outfits.length) $('#outfits').append(element('p', 'Kho dữ liệu chưa có đủ ứng viên cho cả ba vai trò.'));
    $('#results').hidden = false;
    $('#results').scrollIntoView({ behavior: 'smooth' });
    message('Đã tạo gợi ý. Điểm số dùng để xếp hạng trong mô hình.');
  } catch (error) { message(error.message); $('#upload').scrollIntoView({ behavior: 'smooth' }); }
  finally { setBusy(false); }
}
document.querySelectorAll('.pick').forEach(button => button.onclick = () => recommend({ item_id: button.dataset.id }));
let catalogVisible = 12;
function filterCatalog() {
  const text = $('#search').value.toLowerCase().trim();
  let matching = 0;
  document.querySelectorAll('.product').forEach(card => {
    if (!card.dataset.search.includes(text)) { card.hidden = true; return; }
    card.hidden = matching >= catalogVisible;
    matching += 1;
  });
  $('#more-items').hidden = matching <= catalogVisible;
}
$('#search').oninput = () => { catalogVisible = 12; filterCatalog(); };
$('#more-items').onclick = () => { catalogVisible += 12; filterCatalog(); };
filterCatalog();

async function refreshTrainingProgress() {
  try {
    const response = await fetch('/training-status', { cache: 'no-store' });
    if (!response.ok) return;
    const progress = await response.json();
    const labels = { downloading: '↓ ĐANG TẢI DỮ LIỆU', loading: '○ ĐANG NẠP DỮ LIỆU',
      training: '● ĐANG HUẤN LUYỆN', evaluating: '● ĐANG ĐÁNH GIÁ', completed: '● SẴN SÀNG',
      interrupted: '○ ĐÃ TẠM DỪNG', failed: '○ CẦN KIỂM TRA LỖI', pending: '○ CHỜ HUẤN LUYỆN' };
    $('#detector-phase').textContent = labels[progress.phase] || '○ CHỜ HUẤN LUYỆN';
    let detail = progress.message || '';
    if (progress.phase === 'downloading' && progress.download_total_bytes) {
      detail += ' ' + (progress.downloaded_bytes / 1e9).toFixed(2) + ' / ' +
        (progress.download_total_bytes / 1e9).toFixed(2) + ' GB.';
    }
    if (progress.phase === 'training') {
      detail += ` Epoch ${progress.epoch}/${progress.epochs} · batch ${progress.batch}/${progress.batches_per_epoch}`;
      if (Number.isFinite(progress.loss)) detail += ` · loss ${progress.loss.toFixed(3)}`;
      if (Number.isFinite(progress.estimated_remaining_seconds)) {
        detail += ' · còn khoảng ' + (progress.estimated_remaining_seconds / 3600).toFixed(1) + ' giờ (ước tính).';
      }
    }
    if (progress.phase === 'evaluating' && progress.evaluated_images) {
      detail += ` ${progress.evaluated_images}/${progress.valid_images} ảnh.`;
    }
    if (progress.updated_at) detail += ' Cập nhật: ' + new Date(progress.updated_at).toLocaleString('vi-VN') + '.';
    $('#detector-progress').textContent = detail;
  } catch (_) { /* Keep the last known state if the local server is restarting. */ }
}
refreshTrainingProgress();
setInterval(refreshTrainingProgress, 15000);
