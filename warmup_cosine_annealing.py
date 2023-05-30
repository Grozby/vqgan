from typing import Dict

import tensorflow as tf
import math


class WarmUpCosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(
        self,
        warmup_steps: int,
        total_steps: int,
        learning_rate: float,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        # Cosine decay
        learning_rate = (0.5 * self.learning_rate * (1 + tf.cos(
            tf.constant(math.pi) * tf.cast(step - self.warmup_steps, tf.float32)
            / float(self.total_steps - self.warmup_steps))))

        warmup_lr = tf.cast(
            self.learning_rate * (step / self.warmup_steps),
            tf.float32,
        )

        learning_rate = tf.where(
            step < self.warmup_steps,
            warmup_lr,
            learning_rate,
        )

        return tf.where(
            step > self.total_steps,
            0.0,
            learning_rate,
            name="learning_rate",
        )

    def get_config(self) -> Dict:
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
        })
        return config
