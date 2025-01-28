"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
        parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')
        parser.add_argument('--img_path', type=str, default='examples', help='path to test image.')
        # FaceLLM arguments
        parser.add_argument('--cfg', type=str, default=None, help='path to config file.')
        parser.add_argument(
          "--with_tracking",
          action="store_true",
          help="Whether to enable experiment trackers for logging.",
        )
        parser.add_argument(
            "--report_to",
            type=str,
            default="all",
            help=(
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
                "Only applicable when `--with_tracking` is passed."
            ),
        )

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser
