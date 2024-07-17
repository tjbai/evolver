ran a ton of experiments and losing track sort of:

the original motivation was to scale up the UD experiments to the IMDB dataset and beat with longer sequences

the first training run goes poorly and our model underperforms by roughly 4 nats

in the training charts, notice that the index and operation heads hardly decrease. in the original runs we tried to fix this with gradient clipping.

we get a training run with the new loss function and also "fix" the token embedding scaling. this is sup-imdb-[012]

>> insert intermediate conversation with eisner

realize that 1. gradient clipping doesn't work and i wasn't doing what i thought i was doing and 2. our loss function composition is not correct (we average and then sum instead of summing then average)

do more training runs on CLSP grid but we're restricted to small runs. what did we fix? a lot.
- loss composition
- try some scaling