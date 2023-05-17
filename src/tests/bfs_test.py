# bfs tests
import timeit


counter = 0
startTime = timeit.default_timer()
for i in range (0, 100000000):
  counter += 1
print(timeit.default_timer() - startTime)