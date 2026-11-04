obj-m += ps_dma_trial.o

all:
	python3 build_trial.py

clean:
	$(MAKE) -C $(CURDIR)/kernel_build M=$(CURDIR) clean
