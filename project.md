This is the start of CuteAFD - an attention ffn disaggregated LLM engine.

It is born out a few more specific enginer that proceed it

../ds41rt - the premier most optimized one for DeepSeek V4.1 Flash
../ds4rt - trillion parameter model, deep seek v4 pro and several flash variants
../glmrt - glm 5.2 and then 5.3

The principle was one llm, maximally optimized for consumer blackwell (rtx 6000 + many sparks) and later 5090 when it became clear that was possible depending on the model.  Attention on the beefy + compute/high bandwith and exper weights on a pool of sparks.  Sparks started out as 4x, but later supported 2x/6x and maybe 3x, can't remember if we actually did that one.

There is a lot of shared code though, one was forked into the next, into the next.  the first two were developed side by side for a period with continuous back porting of things.  Then there was a pause and another burst of models.  DS 4.1 Flash was a huge winner, and the this engine was forked and deeply optimized for it and it was good.

The ds41rt received the most quality optimizations and bugfix and tests for various things, so it is the canonical.  Yet there are many kernels and at are reusable from the other efforts.  The goal now is to unify them together... the principle is consumer blackwell, super optimized, roce networking between the nodes for real traffic, leverage the mix of compute we have, built in rust for maximum efficiency and strong type checking to support high performance cpu native code while being very safe for llm synthesis of code, tapping the deep library of b12x (sparkinfer) for optimal kernel starting points from the community and extending it as need to fit our use case, extending gptqmodel to support leveraging the same hardware for distributed quantization of huge MooE models to fit the system.  but now having support for many models, not all models, but the ones that really soar.  it could con in the future, but for now it stays opinionated as strong cards as the head, and a pool of sparks.  it will no longer try to specifically load only the exact models we tried it with.  it will try load any directly from the checkpoints.  it may lack the kernels for the configuration, it should say.  it may lack the high speed transformation for loading at nvme rate, it should say.  it can load all from the coordinator and stream into the hosts, i don't really care, because i built sparknest (see ../sparknest) distributed file system to host the vast quantity of models i'm dealing with.  loading just needs to be fast.  anyway in all the situations of not supported, it should output hints looking at the plan for the whole required load to feed to a code agent to be able to complete the loading.  from a model perspective it could mean certain tensors are EXL3 that we didn't develop the kernels for a particular place the model quantized, or whatever.  it avoids blocking, it has a beautiful dashboard, it has an efficient and complete api (today its just openai completions but quite complete including all the constraine decoding and sampling... ds41rt is much much better than the others).

So we build cuteAFD to load these models, run them fast as shit, with the hardware we have 2xRTX 6000 server + 6x sparks.  

We take ds41rt and generalize it, but keep it super fast, and optimized, and make family folders and shared folders for key parts.  we clean the loader planning up and extend our fast loaders to handle generally what we can instead of just exactly what we had, but with no speed compromise.  

Support
DS4 4.1 Flash official
GLM 5.3 EXL3
DS4 Pro / Flash

We have these core quants
deepseek-ai/DeepSeek-V4.1-Flash 
wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1
deepseek-ai/DeepSeek-V4-Flash-0731
wrldsuksgo2mars/DeepSeek-V4-Pro-0813-EXL3-K2-calibrated-v1

Some have interesting speculators, MTP + one best class one is all we care about we already adopted the default best one so use that

There several variations we have two which we can try to support for flexibility we just dont really wanna load the ones that are offloaded from sparknest to models or scratch often because the load speed is slow compared to the nvme grid, but we can use them as test cases as needed during the port

Preserving the excellence of DS4.1Flash
PRobably needing some siginificant work to land GLM and DS4 Pro into the same quality of pipeline tier

Then we extend
GLM 5.3 Flash - zai-org/GLM-5.3-Flash and brandonmusic/GLM-5.3-Flash-tr3-4bpw
Mimo 2.6 Flash - XiaomiMiMo/MiMo-V2-Flash
Mimo 2.6 Pro - XiaomiMiMo/MiMo-V2.6-Pro-RL
Qwen 3.8 Flash - Qwen/Qwen3.8-Flash-Next

There's engram tables... we can use host ram, we can leverage the distributed storage performance to replicate the engram slices across the cluster 
(i wish they all made their quants like DS did with the engram exactly a separate file, maybe they do) MMAP prefetch seems to work really well, but 
it depends on the table size really, so we can consider using sparks memory to hold tables if there lots of extra ram.  these should be options and 
explorations.. for now we just do host ram mem cache, but just letting you know we might play there, we will see where the models go.


mimos are downloading right now others are already here... we will add more models step 5 is coming out minimax has an m3.1 coming, things are moving. we can't keep seprate forks.  we need to ninja design together that lets us keep our max performance and grow with the open sources.


this is what we will do together next my claude.  I have to leave to the maldives tomorrow, so I'm connecting into to you to watch and cheer you on as you do it.  I'll help when I can.  I've prepped the tools for you, the distributed storage grid (sparknest/nest),  super user access (agent-sudo), a first class complete enginer ds41rt, several other engines to think upon how to merge it all together...

we will conquer this and bring the intelligence to everyone!!!!!!!

You can publish direct to gptqmodel and sparkinfer (our b12x fork) as you do this work.  This time lets keep all the performance and documentation litter to a minimum.  very concise measurements and minimal external docs only the code is king, the agent is the builder

vendor those libs and xgrammar we had to patch that shit... etc.. study the repos, make a plan, lets do this, we start with fable and move to our next opus.

voila!

commit and push as you build.  you can invent a new build system if you want... wip and official build and run is pretty good tho, so maybe keep it 
(or refine it).

raptor head coordinator
ostrich,dodo,emu,kiwi,rhea,moa the spark birds

yours to play.  i will cap raptor as 325w per card to keep the system absolutely stable while i'm gone... numbers may be a little harder to reach, mostly prefill i guess.


