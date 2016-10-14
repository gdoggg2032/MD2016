ITER=5
EN=encode.bin
NUM=test.num
all:
	gcc ./train.c -O3 -o ./train
	gcc ./test.c -O3 -o ./test
training:
	./train $(EN) $(NUM) model.init model.trans model.emiss $(ITER)
testing:
	./test model.init model.trans model.emiss test.num pred.num
