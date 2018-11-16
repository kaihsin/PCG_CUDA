CUDA_PATH ?= /usr/local/cuda

CCC ?= g++
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CCC)

INCLUDES := -I$(CUDA_PATH)/samples/common/inc
LIBRARIES :=

## Flags:
NVCCFLAGS := -m64 -g -G
CCFLAGS := -g 
LDFLAGS := 


ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))


SMS ?= 50
GENCODE_FLAGS := -arch=sm_$(SMS)

##############################3


TARGET = pcg32
OBJS = test.o

all : $(TARGET)

test.o : test.cu 
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

$(TARGET): ${OBJS}
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)


.phony: clean

clean:
	rm -f *.o $(TARGET)


