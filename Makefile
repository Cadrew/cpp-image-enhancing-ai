build:
	g++ -std=c++17 -o main.out main.cpp -ltensorflow

run: build
	./main.out