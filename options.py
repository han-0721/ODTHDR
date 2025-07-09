import argparse


class Options:

    #set opt by parser
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # training options
        self.parser.add_argument(
            "--batch_size", 
            type=int, 
            default=1, 
            help="batch size",
        )
        self.parser.add_argument(
            "--epochs", 
            type=int, 
            default=100, 
            help="epochs",
        )
        self.parser.add_argument(
            "--batch_num", 
            type=int, 
            default=100, 
            help="batch num per epochs",
        )

        self.parser.add_argument(
            "--lr", 
            type=float, 
            default=0.0001, 
            help="learning rate",
        )
        self.parser.add_argument(
            "--lr_decay_after",
            type=int,
            default=100,
            help="linear decay of learning rate",
        )

        self.parser.add_argument(
            "--continue_train",
            action="store_true",
            help="load the latest model",
        )
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: 0, 1, 2. use -1 for CPU",
        )

        self.parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        self.parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        self.parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        self.parser.add_argument('--ram_path', type=str, default="./ram_path/ram_swin_large_14m.pth", help='Path to RAM model')
        self.parser.add_argument('--ram_ft_path', type=str, default="./ram_ft_path/DAPE.pth", help='Path to RAM model')

        # diff set args
        self.parser.add_argument(
            "--pretrained_model_name_or_path", 
            default="pretrain_path", 
            type=str
        )

        self.parser.add_argument(
            "--cfg_vsd", 
            default=7.5, 
            type=float
        )
        self.parser.add_argument(
            "--lora_rank", 
            default=8, 
            type=int
        )

        self.parser.add_argument(
            "--dim", 
            default=64, 
            type=int
        )

        self.parser.add_argument(
            "--window_size", 
            default=[4, 4], 
            type=tuple
        )

        self.parser.add_argument(
            "--num_heads", 
            default=4, 
            type=int
        )

        # debugging options
        self.parser.add_argument(
            "--save_ckpt_after",
            type=int,
            default=25,
            help="number of epochs after which checkpoints are saved",
        )
        self.parser.add_argument(
            "--log_after",
            type=int,
            default=800,
            help="number of epochs after which batch, loss is logged",
        )
        self.parser.add_argument(
            "--save_results_after",
            type=int,
            default=400,
            help="number of epochs after which results are saved",
        )

        # testing options
        self.parser.add_argument(
            "--diffhdr_ckpt_path",
            type=str,
            default="../diffhdr.ckpt",
            help="path of diffhdr model checkpoint to be loaded",
        )
        self.parser.add_argument(
            "--mainhdr_ckpt_path",
            type=str,
            default="../mainhdr.ckpt",
            # default="../autodl-tmp/mainhdr/latest_l1_0-1_400.ckpt",
            help="path of mainhdr model checkpoint to be loaded",
        )
        self.parser.add_argument(
            "--log_scores",
            action="store_true",
            help="log PSNR, SSIM scores at evaluation",
        )

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
