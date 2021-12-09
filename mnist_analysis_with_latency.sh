for M in 'lru' 'never-evict' 'random-replace' :
do
        for I in 1 2 3 4 5:
        do
                echo "Iteration $I"
                mkdir -p newMResults_wLatency/$M/run$I

                python MNIST.py --log_path newMResults_wLatency/$M/run$I/ --eviction_method $M --epochs 25 --with_latency
        done
done
