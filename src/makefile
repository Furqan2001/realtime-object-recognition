CXX      = clang++
CXXFLAGS = -std=c++17 -O2 $(shell pkg-config --cflags opencv4) -I../include
LDLIBS   = $(shell pkg-config --libs opencv4)

BINDIR   = ../bin
MODEL    = ../data/resnet18-v2-7.onnx
MODEL_URL = https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx

OBJS = objectRecognition.o processImage.o UnionFind.o features.o objectLearning.o \
       csv_util.o ObjectDB.o EvaluationSystem.o EmbeddingDB.o utilities.o

objectRecognition: $(OBJS)
	@mkdir -p $(BINDIR)
	$(CXX) $^ -o $(BINDIR)/$@ $(LDLIBS)

model: $(MODEL)

$(MODEL):
	@mkdir -p ../data
	curl -L -o $@ $(MODEL_URL)

clean:
	rm -f *.o *~ $(BINDIR)/objectRecognition

.PHONY: model clean
