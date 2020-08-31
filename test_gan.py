import torch
from matplotlib import pyplot as plt
import random


import mndata
import aitools


def loss_mod_f_gen(ys_disc_gen):
    return torch.mean(torch.log(1.0 - ys_disc_gen))


def loss_mod_f_disc(ys_disc_real, ys_disc_gen):
    real = torch.log(ys_disc_real)

    inverted = 1.0 - ys_disc_gen
    gen = torch.log(inverted)

    return -torch.mean(real + gen)


def loss_wasserstein_f_gen(ys_disc_gen):
    return -torch.mean(ys_disc_gen)


def loss_wasserstein_f_disc(ys_disc_real, ys_disc_gen):
    return -torch.mean(ys_disc_real - ys_disc_gen)


"""
loss_f_gen = loss_mod_f_gen
loss_f_disc = loss_mod_f_disc

"""
loss_f_gen = loss_wasserstein_f_gen
loss_f_disc = loss_wasserstein_f_disc


def noise_f(*shape):
    n = 200.0 * torch.rand(shape) - 100.0
    return n


def main():
    images, labels = mndata.load()

    # -------------- HYPERPARAMETERS --------------#

    GENERATOR_NOISE_SIZE = 100
    GENERATOR_HIDDEN_LAYERS = [1200, 1200]
    GENERATOR_HIDDEN_ACTIVATIONS = [torch.nn.functional.relu, torch.nn.functional.relu]
    GENERATOR_LR = 0.000004

    DISCRIMINATOR_HIDDEN_LAYERS = [240, 240]
    DISCRIMINATOR_HIDDEN_ACTIVATIONS = [
        torch.nn.functional.relu,
        torch.nn.functional.relu,
    ]
    DISCRIMINATOR_LR = 0.00005

    TRAINING_DATA_SIZE = 1000
    N_EPISODES = 5000

    DISCRIMINATOR_TRAINING = 1

    # ---------------------------------------------#

    MNIST_SIZE = 28 * 28

    generator_builder = aitools.nn.NetworkFF.Builder(GENERATOR_NOISE_SIZE)
    for layer_size, activation in zip(
        GENERATOR_HIDDEN_LAYERS, GENERATOR_HIDDEN_ACTIVATIONS
    ):
        generator_builder.add_layer(layer_size, activation)
    generator = generator_builder.add_layer(MNIST_SIZE, torch.sigmoid).build()
    optim_generator = torch.optim.Adam(generator.parameters(), lr=GENERATOR_LR)

    discriminator_builder = aitools.nn.NetworkFF.Builder(MNIST_SIZE)
    for layer_size, activation in zip(
        DISCRIMINATOR_HIDDEN_LAYERS, DISCRIMINATOR_HIDDEN_ACTIVATIONS
    ):
        discriminator_builder.add_layer(layer_size, activation)
    discriminator = discriminator_builder.add_layer(1, torch.sigmoid).build()
    optim_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=DISCRIMINATOR_LR
    )

    data_training = images[:TRAINING_DATA_SIZE]

    losses_discriminator = []
    losses_generator = []
    try:
        for ep in range(N_EPISODES + 1):
            # --------- DISCRIMINATOR TRAIN --------- #

            loss_sum = 0.0
            for _ in range(DISCRIMINATOR_TRAINING):
                ys_disc_real = discriminator(data_training)

                noise_data = noise_f(TRAINING_DATA_SIZE, GENERATOR_NOISE_SIZE)
                ys_gen = generator(noise_data)
                ys_disc_gen = discriminator(ys_gen)

                optim_discriminator.zero_grad()
                loss = loss_f_disc(ys_disc_real, ys_disc_gen)
                loss_sum += loss
                loss.backward()
                optim_discriminator.step()
            losses_discriminator.append(loss_sum / float(DISCRIMINATOR_TRAINING))

            # --------- GENERATOR TRAIN --------- #
            loss_sum = 0.0

            noise_data = noise_f(TRAINING_DATA_SIZE, GENERATOR_NOISE_SIZE)
            ys_gen = generator(noise_data)
            ys_disc_gen = discriminator(ys_gen)

            optim_generator.zero_grad()
            loss = loss_f_gen(ys_disc_gen)
            loss_sum = loss
            loss.backward()
            optim_generator.step()

            losses_generator.append(loss_sum)

            print(
                "{} ->\tdisc:[{:.2f}]\tgen:[{:.2f}]".format(
                    ep, losses_discriminator[-1], losses_generator[-1]
                )
            )
    except KeyboardInterrupt:
        pass

    plt.plot(losses_generator, "g")
    plt.plot(losses_discriminator, "r")
    plt.show(block=False)

    iv = aitools.utils.ImageVisual(28, 28, 3)
    validation = [generator(noise_f(GENERATOR_NOISE_SIZE)) for _ in range(10)]
    random.shuffle(data_training)
    for t, v in zip(data_training[: len(validation)], validation):
        print("{} --- {}".format(discriminator(t).item(), discriminator(v).item()))

    iv.display(validation)
    input()
    iv.close()


if __name__ == "__main__":
    main()
