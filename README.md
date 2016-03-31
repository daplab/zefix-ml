# Zefix Machine Learning

Machine learning components (topic modeling) for the zefix-notifier project.
Latent Dirichlet Allocation implementation of MLib on Zefix company descriptions.


## How to run it

Build and push it to the DAPLAB

```
mvn clean install && scp target/zefix-ml-1.0.0-SNAPSHOT.jar pubgw1.daplab.ch:
```

Run it

```
spark-submit --master yarn --num-executors 4 --class org.daplab.zefix.lda.topics zefix-ml-1.0.0-SNAPSHOT.jar

--conf spark.rpc.lookupTimeout=6000
--conf spark.rpc.askTimeout=6000
```

(note that the number of executors should be fine tuned)