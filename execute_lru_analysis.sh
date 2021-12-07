for I in 1 2 3 4 5 6 7 8 9 10:
do
        echo "Iteration $I"
        mkdir -p lru/run$I
        python3.7 CreditCardFraudNN.py --log_path lru/run$I/ --eviction_method lru --epochs 25
done
