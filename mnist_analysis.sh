for M in 'lru' 'never-evict' 'random-replace' :
do
        for I in 1 2 3 4 5:
        do
                echo "Iteration $I"
                mkdir -p new_MResults/$M/run$I

                python MNIST.py --log_path new_MResults/$M/run$I/ --eviction_method $M --epochs 25

        done
done
