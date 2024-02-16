import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot


# Disable eager execution to ensure compatibility with python3.11+/tensorflow 2
tf.compat.v1.disable_eager_execution()


# Download Google Billion Word at http://www.statmt.org/lm-benchmark/ and
# fill in the path to the extracted files here!
cur_path = os.getcwd()
DATA_DIR = os.path.join(cur_path,'seq')

BATCH_SIZE = 32 # Batch size
ITERS = 1000 # How many iterations to train for
SEQ_LEN = 50 # Sequence length in characters
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 5 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 14098 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).

lib.print_model_settings(locals().copy())

lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=SEQ_LEN,
    max_n_examples=MAX_N_EXAMPLES,
    data_dir=DATA_DIR
)

def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, len(charmap)])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random.normal(shape) # change to tf.random.normal for tf2.x

def ResBlock(name, inputs, sf=0.3):
    '''
    Defines what as known as a 'residual block' from the original ResNet architecture.

    Idea is to use 'skip connections' where you add the input of the network x to the output h(x)
    so the hidden layers will have to model the residual quantity h(x) - x.

    @param name: str, prefix for the 1D convolutional layers within the ResBlock.
    @param inputs: input tensor
    @param sf: scaling factor used to control the influence of the residual connection h(x) - x.

    @return h(x), activation output of the block but formed using the shortcut connections.
    
    @reference: He et al. (2015). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
    
    '''

    # Add the input tensor to the output tensor
    output = inputs

    # The architecture of a Residual unit is just two convolutional layers with ReLU activation.
    # Layer 1
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.1', DIM, DIM, 5, output)

    # Layer 2
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name+'.2', DIM, DIM, 5, output)

    # model h(x) = x + (h(x) - x)
    return inputs + (sf * output)


def Generator(n_samples, noise_dim, seq_len=SEQ_LEN, output_dim=DIM, n_resblocks=5):
    '''
    Defines the generator component of a GAN which converts Gaussian noise vector to a synthetic sample.

    @param n_samples: int, number of samples in the gaussian noise distribution.
    @param noise_dim: int, dimensionality of input noise
    @param seq_len: int, sequence length
    @param output_dim: int, output dimensionality
    @param n_resblocks: int, number of resblocks in the generator.
    @param n_classes: int, number of classes in the output
    
    @return output: softmax activation 
    '''

    output = make_noise(shape=[n_samples, noise_dim])

    # Linear transform/fully connected layer (output = Wx + b; torch.nn.Linear in PyTorch)
    output = lib.ops.linear.Linear('Generator.Input', noise_dim, seq_len * output_dim, output)
    output = tf.reshape(output, [-1, output_dim, seq_len])

    # Generator architecture consists of (default) 5 residual units
    for i in range(n_resblocks):
        output = ResBlock(f'Generator.{i+1}', output)

    # 1-D Convolutional layer to generate the synthetic data output.
    # len(charmap) is the number of output classes, (A/C/T/G) or 4.
    # At this point the output tensor is [output_dim, n_characters, seq_len]
    output = lib.ops.conv1d.Conv1D('Generator.Output', output_dim, len(charmap), 1, output)  

    # Transpose, apply softmax
    output = tf.transpose(output, [0, 2, 1]) # Permutation indices; 1 corresponds to the seq length
    output = softmax(output)
    return output

def Discriminator(inputs):
    output = tf.transpose(inputs, [0,2,1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', SEQ_LEN*DIM, 1, output)
    return output

real_inputs_discrete = tf.compat.v1.placeholder(tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
fake_inputs = Generator(BATCH_SIZE)
fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims-1)

disc_real = Discriminator(real_inputs) 
disc_fake = Discriminator(fake_inputs)

disc_cost = tf.compat.v1.reduce_mean(disc_fake) - tf.compat.v1.reduce_mean(disc_real)
gen_cost = -tf.compat.v1.reduce_mean(disc_fake)



# WGAN lipschitz-penalty
alpha = tf.random.uniform( # change to tf.random.uniform
    shape=[BATCH_SIZE,1,1], 
    minval=0.,
    maxval=1.
)
differences = fake_inputs - real_inputs
interpolates = real_inputs + (alpha*differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.compat.v1.math.sqrt(tf.compat.v1.reduce_sum(tf.compat.v1.math.square(gradients), reduction_indices=[1,2]))   # Update to tf 2.x
gradient_penalty = tf.compat.v1.reduce_mean((slopes-1.)**2)
disc_cost += LAMBDA*gradient_penalty

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

gen_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
disc_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

saver = tf.compat.v1.train.Saver(max_to_keep=4)
# Dataset iterator
def inf_train_gen():
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]], 
                dtype='int32'
            )

			
# During training we monitor JS divergence between the true & generated ngram
# distributions for n=1,2,3,4. To get an idea of the optimal values, we
# evaluate these statistics on a held-out set first.
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[10*BATCH_SIZE:], tokenize=False) for i in range(4)]
validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines[:10*BATCH_SIZE], tokenize=False) for i in range(4)]
for i in range(4):
    print("validation set JSD for n={}: {}".format(i+1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
true_char_ngram_lms = [language_helpers.NgramLanguageModel(i+1, lines, tokenize=False) for i in range(4)]

with tf.compat.v1.Session() as session: # compat.v1

    session.run(tf.compat.v1.global_variables_initializer()) # update to tf.compat.v1

    def generate_samples():
        samples = session.run(fake_inputs)
        samples = np.argmax(samples, axis=2)
        decoded_samples = []
        for i in range(len(samples)):
            decoded = []
            for j in range(len(samples[i])):
                decoded.append(inv_charmap[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    gen = inf_train_gen()
    

    for iteration in range(ITERS):
        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        for i in range(CRITIC_ITERS):
            _data = next(gen)  # Use next(gen) instead of gen.next() in tf2.x
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_inputs_discrete:_data}
            )

        lib.plot.plot('time', time.time() - start_time)
        lib.plot.plot('train disc cost', _disc_cost)

        if iteration % 100 == 99:
            saver.save(session, './my-model', global_step=iteration)
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            with open('samples_{}.txt'.format(iteration), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")
        if iteration % 100 == 99:
            samples = []
            for i in range(10):
                samples.extend(generate_samples())

            for i in range(4):
                lm = language_helpers.NgramLanguageModel(i+1, samples, tokenize=False)
                lib.plot.plot('js{}'.format(i+1), lm.js_with(true_char_ngram_lms[i]))

            with open('samples_{}.txt'.format(iteration), 'w') as f:
                for i,s in enumerate(samples):
                    s = "".join(s)
                    f.write('>' + str(i) + '\n')
                    f.write(s + "\n")

        if iteration % 100 == 99:
            lib.plot.flush()
        
        lib.plot.tick()
