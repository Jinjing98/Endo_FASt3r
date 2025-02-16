from __future__ import absolute_import, division, print_function

from trainer_end_to_end_reloc3r_no_amp_from_scratch import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    print("NO AMP!!!FROM SCRATCH")
    trainer = Trainer(opts)
    trainer.train()
