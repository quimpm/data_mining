# Deep Reinforcement Learning - Car Race

Even though I haven't achived to solve the problem. I will describe how I tryed to takle it.

## Car Racing Problem

This problem had a State of a 96x96x3 matrix representing an RGB image of 96x96 pixels. In order to make the state fittable to our QNet's, this state required a transofrmation (flatten), as linear networks need to a 1d input.

Also, the state can be treated following two ways. Continuous or Discrete. With discrete is much more easyer as we will have an output of 5 possible actions. Therfore, the Enviroment is created using the Discrete method.

Next I will to describe the iterations that I did on trying to solve this problem.

## First iteration

In the first iteration, the State is flated into an array of 96x96X3=27648. I tried to fit the networks using this huge state (For fitting this state the BatchSize has needed to be modified to fit the memory). But this didn't work out.

The first thing that came to my mind was that the state was super big and there were not enough neurons. So I increased the number of the liear neurons and tried again. Again didn't work out. 

## Second Iteration

As increasing the number of neurons was not succesfull, I though on some ways of doing the state a bit smaller. Searching in internet I found that maybe, a god way of going it was by grayscaling the image. Therfore, instead of an array of 96x96x3, the state will be represented in an array of 96X96x1=**9216**. Still a big state but three times lesser that the beggining one.

Tried the same aproach as before. I executed the problem with less and more neurons in the Linear Layers with no success.

Also, due to the enormous state the Episodes were running super slow. 

## Third iteration

At this point I thought that the network arch was not enough to handle the enormus problem. So I developed a new class called ConvNN containing an implementation of a Convolutional network. Once implemented, i tried to execute it with the same result... (Also, some addaptations had to be done to at agent level to integrate the new model ConvNN)

In this case I used a convolutional network of 3 channels, so no more grayscale transformation at this point. 

A think to note is that the Episodes where a bit slow also... Maybe a bit more that with the Linear model.

## Fourth iteration

Finally, I tried to finetune the epsilon by setting more randomnes at the beggining(more exploration) and fine tunning this epsilon in terms of the scores. This was done becouse it was thought that the agent was falling into some Local Minima.

## Final thoughts

After leaving varelly the whole Sunday running the program, I found that the agent was starting to learn with the last aproach appliying ConvNN and Epsioln fine tunning. But I ran Out of time! :(. In the notebook you can find the last scores after 509 episodes. Even seems the agent found a local minima. 

Maybe a bit more of work on reward or in Local minima avoiding would have been the last piece to make the whole engine work.