build:
	g++ -std=c++17 -o main.out main.cpp -ltensorflow -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -ltensorflow_framework

run: build
	./main.out