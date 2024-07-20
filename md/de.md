in step 1:
BOS and EOS will both correspond to memory (doesn't matter for EOS)
-> during inference initiate with mem[0]

tgt_pad_mask becomes source_pad_mask in the next step
the new tgt_pad_mask is always dynamically computed based on which children we attend over

concern: BOS position will always be masked out?
solution: set BOS parents to -2 always (!= -1)

in step 2:
BOS will correspond to memory again (mem + torch.where(add_par, permuted_mem, 0) -> mem because add_par = False)

in step 3, 4, 5 (tok:rel:pos prediction):
BOS corresopnds to memory (me + torch.where(add_rel, self.embedding(tgt_rel), 0) -> mem because add_rel = False)

     VERB
    /    \
SUBJECT   OBJECT

"She ate food" corresponds to:
-> ate/blank/VERB
-> INS CPY(1) INS
-> INS(2) (ate/blank/VERB) INS(2)
-> INS(2, nsubj) (ate/blank/VERB) INS(2, obj)
-> INS(2, nsubj, PRP) (ate/blank/VERB) INS(2, obj, NN)
-> INS(2, nsubj, PRP, she) (ate/blank/VERB) INS(2, obj, NN, food)

In terms of embeddings (ignoring positionals), at each step we reencode then decode autoregressively:
-> s (x0) /s
-> s' (PLH x0') (PLH) /s
-> (PLH+x0'') (x0''+DONE) (PLH+x0'')
-> 