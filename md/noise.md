for the complete architecture we also need to define what the noising process looks like...

our model is parametrized by the form of the latent distribution

so far:
- p drop noise to check whether sharing embeddings/evolver architecture can learn simple patterns and use deep embeddings
- random dep noise, generally leads to poor performance and difficult for model to jointly learn where to insert and what

ideas:

the model should have sufficient expressive power to generate all target sequences in a single step. our theorized improvement is because multiple steps allows the model to encode deep embeddings of reused tokens. 

are we hamstringing our model by _preventing_ it from generating complete sequences at each step? does a better latent distribution model the process of revision more closely?
