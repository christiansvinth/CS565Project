for M in 'lru' 'never-evict' 'importance' 'random-replace' :
do
        for I in 1 2 3 4 5:
        do
                echo "Iteration $I"
                mkdir -p CCResults/$M/run$I
                mkdir -p MResults/$M/run$I

                python CreditCardFraudNN.py --log_path CCResults/$M/run$I/ --eviction_method $M --epochs 15
                python MNIST.py --log_path MResults/$M/run$I/ --eviction_method $M --epochs 25
        done
done
