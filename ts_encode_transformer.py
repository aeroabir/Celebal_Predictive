from transformer_models_ts import (create_padding_mask,
                                   create_look_ahead_mask,
                                   Transformer)

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import pickle
import sys
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm import trange


def create_masks(inp, tar):
    # Encoder padding mask
    # (batch_size, 1, 1, seq_length, num_features)
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    # (batch_size, 1, 1, seq_length, num_features)
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input
    # received by the decoder.
    # (seq_len, seq_len)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

    # dec_target_padding_mask = create_padding_mask(tar)
    # combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, look_ahead_mask, dec_padding_mask


def evaluate_one_example(encoder_input, decoder_input, transformer):

    encoder_input = tf.expand_dims(encoder_input, 0)  # (1, seq_len, features)
    output = tf.expand_dims(decoder_input, 0)  # (1, features)
    output = tf.expand_dims(output, 0)  # (1, 1, features)
    # print(encoder_input.shape, decoder_input.shape, output.shape)
    MAX_LENGTH = encoder_input.shape[1]

    for i in range(MAX_LENGTH-1):
        # predictions.shape == (batch_size, seq_len, vocab_size)
        # print(i, encoder_input.shape, output.shape)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     None,
                                                     None,
                                                     None)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, features)
        output = tf.concat([output, predictions], axis=-2)  # concat on sequence

    return output, attention_weights


def get_ae_features(encoder_input, transformer, embed_dim):

    num_points, seq_len, num_features = encoder_input.shape
    # output = tf.zeros(shape=(num_points, seq_len, embed_dim))
    output_list = []
    if num_points > 100:
        batch_size = 64
        dataset = tf.data.Dataset.from_tensor_slices((encoder_input))
        dataset = dataset.batch(batch_size)
        for batch, inp in enumerate(dataset):
            out = transformer.encoder(inp, training=False, mask=None)
            # i0 = batch_size * batch
            # i1 = i0 + batch_size
            output_list.append(out)
        output = tf.concat(output_list, axis=0)
    else:
        output = transformer.encoder(encoder_input, training=False, mask=None)

    return output


def train(X, Y, checkpoint_path, **kwargs):

    # typical example, num_layers=2, d_model=512, num_heads=8, dff=2048,
    # input_vocab_size=8500, target_vocab_size=8000,
    # pe_input=10000, pe_target=6000
    num_layers = kwargs.get("num_layers", 2)
    d_model = kwargs.get("d_model", 128)
    num_heads = kwargs.get("num_heads", 8)
    dff = kwargs.get("dff", 128)
    BATCH_SIZE = kwargs.get("BATCH_SIZE", 64)
    epochs = kwargs.get("epochs", 100)
    action = kwargs.get("action", "train")
    padding_reqd = kwargs.get("padding_reqd", False)
    normalize_flag = kwargs.get("normalize_flag", True)
    split = kwargs.get("split", True)
    seed = kwargs.get("seed", 100)

    xpoints, inp_seq_len, inp_features = X.shape
    ypoints, out_seq_len, out_features = Y.shape
    pe_input = inp_seq_len
    pe_target = out_seq_len
    target_vocab_size = out_features

    if split:
        print("Splitting the data: ")
        X_train,  X_test, Y_train, Y_test = train_test_split(X, Y,
                                                             test_size=0.2,
                                                             random_state=seed)
    else:
        X_train, Y_train = X, Y
    print("Train and Test data shape:", X_train.shape, X_test.shape)
    BUFFER_SIZE = len(X_train)
    steps_per_epoch = BUFFER_SIZE//BATCH_SIZE

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    # val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    # test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

    # sample for testing
    sample_transformer = Transformer(num_layers=num_layers,
                                     d_model=d_model,
                                     num_heads=num_heads, dff=dff,
                                     target_vocab_size=out_features,
                                     pe_input=inp_seq_len,
                                     pe_target=out_seq_len)

    temp_input = tf.random.uniform((BATCH_SIZE, inp_seq_len, inp_features),
                                   dtype=tf.float32, minval=0, maxval=1)
    temp_target = tf.random.uniform((BATCH_SIZE, out_seq_len, out_features),
                                    dtype=tf.float32, minval=0, maxval=1)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)
    print("Output shape:", fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)

    # d_model % self.num_heads == 0
    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              target_vocab_size, pe_input, pe_target, rate=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                         beta_2=0.999, epsilon=1e-7)
    loss_object = tf.keras.losses.MeanSquaredError(
                    reduction=tf.keras.losses.Reduction.NONE,
                    name='mean_squared_error')

    def loss_function(real, pred):
        loss_ = loss_object(real, pred)  # batch X seq_len
        loss_ = tf.reduce_sum(loss_, axis=-1)  # batch
        return tf.reduce_mean(loss_)

    def accuracy_function(real, pred):
        mae = tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.NONE)
        loss_ = mae(real, pred)
        loss_ = tf.reduce_sum(loss_, axis=-1)
        return tf.reduce_mean(loss_)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=5)
    checkpoint_directory = os.path.dirname(checkpoint_path)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    if action == "train":
        # The @tf.function trace-compiles train_step into a TF graph for faster
        # execution. The function specializes to the precise shape of the argument
        # tensors. To avoid re-tracing due to the variable sequence lengths or variable
        # batch sizes (the last batch is smaller), use input_signature to specify
        # more generic shapes.

        train_step_signature = [
            tf.TensorSpec(shape=(None, None, inp_features), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, out_features), dtype=tf.float32),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1, :]
            tar_real = tar[:, 1:, :]

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            enc_padding_mask = None
            dec_padding_mask = None

            with tf.GradientTape() as tape:
                predictions, _ = transformer(inp, tar_inp,
                                            True,
                                            enc_padding_mask,
                                            combined_mask,
                                            dec_padding_mask)
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)    
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

        train_dataset = train_dataset.batch(BATCH_SIZE)
        print(f"Training on {BUFFER_SIZE} data points over {epochs} epochs")
        t = trange(epochs, desc='Epoch Desc', leave=True)
        for epoch in t:
            start = time.time()

            train_loss.reset_states()
            train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(train_dataset):

                # check for dimensions
                # print(inp.shape, tar.shape)
                # print(inp.dtype, tar.dtype)
                # tar_inp = tar[:, :-1, :]
                # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
                # predictions, _ = transformer(inp, tar_inp, True, None, combined_mask, None)
                # print(predictions.shape)

                train_step(inp, tar)

                # if batch % 50 == 0:
                #     print('Epoch {} Batch {} MSE-Loss {:.4f} MAE-Loss {:.4f}'.format(
                #         epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            t.set_description('Epoch {} MSE-Loss {:.6f}  MAE-Loss {:.6f}'.format(epoch + 1,
                                                                train_loss.result(),
                                                                train_accuracy.result()))
            t.refresh()
            if (epoch + 1) % 5 == 0:
                ckpt_save_path = ckpt_manager.save()
                # print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

            # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
            #                                             train_loss.result(), 
            #                                             train_accuracy.result()))

            # print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        print("Training Complete!")

    elif action == "predict_all":

        pred = get_ae_features(X, transformer, d_model)
        return pred

    elif action == "predict_batch":
        # ckpt.restore(checkpoint_path).expect_partial()
        dataset_test = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(BATCH_SIZE, drop_remainder=False)
        example_X, example_Y = next(iter(dataset_test))
        print(example_X.shape, example_Y.shape)
        predictions = np.zeros((BATCH_SIZE, out_seq_len, out_features), dtype=np.float32)
        for index in trange(BATCH_SIZE):
            enc_inp, dec_inp0 = example_X[index, :, :], example_Y[index, 0, :]
            pred_ii, _ = evaluate_one_example(enc_inp, dec_inp0, transformer)
            predictions[index, :, :] = pred_ii
            # predictions = np.append(predictions, pred_seq, axis=0)

        nsample = 5
        len_x, len_y = example_X.shape[1], example_Y.shape[1]
        shifted_x = [x+len_x for x in range(len_y)]
        # indices = random.sample([ii for ii in range(example_X.shape[0])], k=nsample)
        for ii in range(nsample):
            filename = "image_"+str(ii)+".png"
            plt.figure(ii+1)
            cnum = random.randint(0, len(example_X)-1)
            colnum = random.randint(0, out_features-1)
            print(ii, cnum, colnum)
            # input
            plt.plot(example_X[cnum, :, colnum])

            # target
            plt.plot(shifted_x, example_Y[cnum, :, colnum])  # , 'bo--'

            # prediction
            plt.plot(shifted_x, predictions[cnum, :, colnum])  # , 'r+'
            plt.savefig(os.path.join(checkpoint_directory, filename))
            plt.title("Feature-"+str(colnum))
            print(f"saved {filename}")

    else:
        return None


if __name__ == "__main__":

    data_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/abir-msft/code/Abir/'
    ckpt_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/abir-msft/code/Abir/training_transformer_all/cp.ckpt"
    # model_path = "/home/AzureUser/notebooks/time-series-classification/training_transformer/ckpt-53"
    num_features = 3

    with open(os.path.join(data_dir, 'autoencoder-data-v01.pkl'), 'rb') as fr:
        X, Y = pickle.load(fr)

    # X, Y = np.array(X_s), np.array(Y_s)
    print(X.shape, Y.shape)

    xmax = np.amax(X, axis=(0, 1))
    print(xmax)

    ymax = np.amax(Y, axis=(0, 1))
    print(ymax)

    X = X / xmax
    Y = Y / ymax

    sys.exit()
    # X_small = X[:, :, 0:num_features]
    # print(X_small.shape)

    train(X,
          X,
          ckpt_path,
          num_layers=2,
          d_model=128,
          epochs=10,
          BATCH_SIZE=32,
          action="train",
          )
    # train(X_small, X_small, ckpt_path, BATCH_SIZE=64, train_flag=False)

