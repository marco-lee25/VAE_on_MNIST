## AE
The main objective of the AE is to map/compress an input X to a lower/latent dimension $Z$ through an encoder and reconstruct it using the decoder. 
In general, AE is learning a representation of the original input.
![image](https://github.com/marco-lee25/AI_NoteBook/assets/72645115/18323b3f-3313-454b-bb0b-0451ab046387)

## VAE
VAE stands for variational autoencoder, the same as the basic VE, it has an encoder and decoder.
While comparing to the basic AE, the encoder of VAE instead outputs the means and covariance matrices of a multivariate normal distribution where all of the dimensions are independent, other than a code.

You can think of the output of the encoder of a VAE this way: the means and standard deviations of a set of independent normal distributions, with one normal distribution (one mean and standard deviation) for each latent dimension. 
![image](https://github.com/marco-lee25/AI_NoteBook/assets/72645115/7a356f29-e2f4-4471-83b6-1a62d9e7a15e)

The workflow of the VAE toward the MNIST dataset is as follows:
1. Input an image to the encoder
2. Mean and SD and output by the encoder
3. Sample from a distribution with the output mean and SD
4. Input sampled value (vector/latent) as the input to the decoder
5. Generate a fake sample
6. Backpropagate using reconstruction loss between real input and fake output.

## ELBO
ELBO stand for evidence lower bound, while training a VAE, we are trying to maximize the likelihood of the real images, in which we'd like the learned probability distribution to think it's likely that a real image. while finding this likelihood is mathematically intractable. Instead, we can get a good lower bound on the likelihood, meaning we can figure out what the worst-case scenario of the likelihood is and maximize it instead (maximize lower bound => making the likelihood better).

#### ELBO for VAE : 
$\mathbb{E}\left(\log p(x|z)\right) + \mathbb{E}\left(\log \frac{p(z)}{q(z)}\right)$
which is equivalent to 
$\mathbb{E}\left(\log p(x|z)\right) - \mathrm{D_{KL}}(q(z|x)\Vert p(z))$

And we can break this formula down into 2 parts
1. Reconstruction loss : $\mathbb{E}\left(\log p(x|z)\right)$ 
2. KL divergence term : $\mathrm{D_{KL}}(q(z|x)\Vert p(z))$

 ## Result 
 #### Epoch 0
 ![image](https://github.com/marco-lee25/AI_NoteBook/assets/72645115/cf3db5e9-206f-4f31-b455-1f3535638bc6)
 #### Epoch 9
 ![image](https://github.com/marco-lee25/AI_NoteBook/assets/72645115/894dc8b3-4f79-4c05-88ce-dff3f1d758a6)





##### Reference
Coursera course "Generative Adversarial Networks (GANs) Specialization". (https://www.coursera.org/learn/build-better-generative-adversarial-networks-gans/) 
It is a very great course introducing the different GAN models, do check it out if you want a deeper understanding of the above content.

深度學習Paper系列(04)：Variational Autoencoder (VAE) : https://tomohiroliu22.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92paper%E7%B3%BB%E5%88%97-04-variational-autoencoder-vae-a7fbc67f0a2

AutoEncoder (一)-認識與理解 : https://medium.com/ml-note/autoencoder-%E4%B8%80-%E8%AA%8D%E8%AD%98%E8%88%87%E7%90%86%E8%A7%A3-725854ab25e8


