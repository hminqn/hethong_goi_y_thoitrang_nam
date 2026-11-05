"""Download the full official data, then train all images for ten epochs."""
import argparse
import sys
import traceback

from .detector_training import run, status
from .fashionpedia_setup import prepare
from .train import seed_everything


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    status(phase='downloading', message='Đang tải và kiểm tra toàn bộ Fashionpedia (4,14 GB). Sau đó tự động huấn luyện 10 epoch.')
    try:
        report = prepare()
    except Exception as error:
        status(phase='failed', message='Tải/kiểm tra dữ liệu chưa thành công: ' + str(error), traceback=traceback.format_exc())
        raise
    splits = report['splits']
    seed_everything()
    args = argparse.Namespace(train_annotations=splits['train']['annotation_file'],
                              train_images=splits['train']['image_directory'],
                              valid_annotations=splits['val']['annotation_file'],
                              valid_images=splits['val']['image_directory'],
                              epochs=10, batch_size=1, min_size=512, max_size=768,
                              checkpoint_every=100, resume=True)
    run(args)


if __name__ == '__main__':
    main()
