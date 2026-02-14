"""Tests for training loop components: loss, IoU, train_step, eval_step."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from models import UNet
from train import compute_iou, dice_loss, eval_step, train_step

# Use small spatial size (64x64) for fast tests
SIZE = 64
NUM_CLASSES = 6
BATCH = 2


@pytest.fixture
def model():
    m = UNet(num_classes=NUM_CLASSES, rngs=nnx.Rngs(0))
    m.train()
    return m


@pytest.fixture
def optimizer(model):
    return nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


@pytest.fixture
def dummy_batch():
    images = jnp.ones((BATCH, SIZE, SIZE, 1))
    labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
    return images, labels


# ── dice_loss ────────────────────────────────────────────────────────────────


class TestDiceLoss:
    def test_returns_scalar(self):
        logits = jnp.zeros((BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        loss = dice_loss(logits, labels, NUM_CLASSES)
        assert loss.shape == ()

    def test_perfect_prediction_has_low_loss(self):
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        logits = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -10.0)
        logits = logits.at[:, :, :, 0].set(10.0)
        loss = float(dice_loss(logits, labels, NUM_CLASSES))
        assert loss < 0.1

    def test_wrong_prediction_has_higher_loss_than_correct(self):
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        # Correct prediction
        logits_correct = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -10.0)
        logits_correct = logits_correct.at[:, :, :, 0].set(10.0)
        # Wrong prediction
        logits_wrong = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -10.0)
        logits_wrong = logits_wrong.at[:, :, :, 1].set(10.0)
        loss_correct = float(dice_loss(logits_correct, labels, NUM_CLASSES))
        loss_wrong = float(dice_loss(logits_wrong, labels, NUM_CLASSES))
        assert loss_wrong > loss_correct

    def test_loss_in_zero_one_range(self):
        logits = jax.random.normal(jax.random.key(0), (BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jax.random.randint(jax.random.key(1), (BATCH, SIZE, SIZE), 0, NUM_CLASSES)
        loss = float(dice_loss(logits, labels, NUM_CLASSES))
        assert 0.0 <= loss <= 1.0

    def test_loss_is_nonnegative(self):
        logits = jax.random.normal(jax.random.key(0), (BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        assert float(dice_loss(logits, labels, NUM_CLASSES)) >= 0.0

    def test_uniform_logits_give_high_loss(self):
        """Uniform predictions (1/C per class) should yield a poor dice score."""
        logits = jnp.zeros((BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        loss = float(dice_loss(logits, labels, NUM_CLASSES))
        assert loss > 0.3


# ── compute_iou ──────────────────────────────────────────────────────────────


class TestComputeIoU:
    def test_returns_per_class_array(self):
        logits = jnp.zeros((BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        ious = compute_iou(logits, labels, NUM_CLASSES)
        assert ious.shape == (NUM_CLASSES,)

    def test_perfect_prediction_gives_iou_1(self):
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        logits = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -100.0)
        logits = logits.at[:, :, :, 0].set(100.0)
        ious = compute_iou(logits, labels, NUM_CLASSES)
        assert float(ious[0]) == pytest.approx(1.0)

    def test_absent_class_gives_nan(self):
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)
        logits = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -100.0)
        logits = logits.at[:, :, :, 0].set(100.0)
        ious = compute_iou(logits, labels, NUM_CLASSES)
        for c in range(1, NUM_CLASSES):
            assert jnp.isnan(ious[c])

    def test_no_overlap_gives_iou_0(self):
        labels = jnp.ones((BATCH, SIZE, SIZE), dtype=jnp.int32)
        logits = jnp.full((BATCH, SIZE, SIZE, NUM_CLASSES), -100.0)
        logits = logits.at[:, :, :, 0].set(100.0)
        ious = compute_iou(logits, labels, NUM_CLASSES)
        assert float(ious[0]) == pytest.approx(0.0)
        assert float(ious[1]) == pytest.approx(0.0)

    def test_iou_values_in_valid_range(self):
        logits = jax.random.normal(jax.random.key(1), (BATCH, SIZE, SIZE, NUM_CLASSES))
        labels = jax.random.randint(jax.random.key(2), (BATCH, SIZE, SIZE), 0, NUM_CLASSES)
        ious = compute_iou(logits, labels, NUM_CLASSES)
        valid = ious[~jnp.isnan(ious)]
        assert jnp.all(valid >= 0.0)
        assert jnp.all(valid <= 1.0)


# ── train_step ───────────────────────────────────────────────────────────────


class TestTrainStep:
    def test_returns_loss_and_logits(self, model, optimizer, dummy_batch):
        images, labels = dummy_batch
        loss, logits = train_step(model, optimizer, images, labels)
        assert loss.shape == ()
        assert logits.shape == (BATCH, SIZE, SIZE, NUM_CLASSES)

    def test_loss_is_finite(self, model, optimizer, dummy_batch):
        images, labels = dummy_batch
        loss, _ = train_step(model, optimizer, images, labels)
        assert jnp.isfinite(loss)

    def test_parameters_change_after_step(self, model, optimizer, dummy_batch):
        images, labels = dummy_batch
        _, state_before = nnx.split(model)
        params_before = jax.tree.leaves(state_before)
        first_param_before = params_before[0].copy()

        train_step(model, optimizer, images, labels)

        _, state_after = nnx.split(model)
        params_after = jax.tree.leaves(state_after)
        first_param_after = params_after[0]

        assert not jnp.allclose(first_param_before, first_param_after)

    def test_loss_decreases_over_steps(self, model, optimizer):
        images = jnp.ones((BATCH, SIZE, SIZE, 1))
        labels = jnp.zeros((BATCH, SIZE, SIZE), dtype=jnp.int32)

        model.train()
        loss_first, _ = train_step(model, optimizer, images, labels)

        for _ in range(10):
            train_step(model, optimizer, images, labels)

        loss_later, _ = train_step(model, optimizer, images, labels)
        assert float(loss_later) < float(loss_first)


# ── eval_step ────────────────────────────────────────────────────────────────


class TestEvalStep:
    def test_returns_logits(self, model):
        model.eval()
        images = jnp.zeros((BATCH, SIZE, SIZE, 1))
        logits = eval_step(model, images)
        assert logits.shape == (BATCH, SIZE, SIZE, NUM_CLASSES)

    def test_deterministic_in_eval_mode(self, model):
        model.eval()
        images = jnp.ones((BATCH, SIZE, SIZE, 1))
        out1 = eval_step(model, images)
        out2 = eval_step(model, images)
        assert jnp.allclose(out1, out2)
