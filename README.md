## Getting started Java benchmark

Getting started code for the Evotor competition on Boosters (Java)

- this is the getting started code for the evotor competition on Boosters (in Russian: https://boosters.pro/champ_evotor)
- eligible submissions for this competitions must be created with Java code, which is a bit unusual requirement for Data Science competitions
- so to help you get started I created a simple benchmark
- it can get 0.829305 (which was the 1st place at the moment of creating it)

Some details:

- the most interesting class is `evotor.GettingStarted`
- first it reads the data and tokenizes the titles
- then the titles are vectorizes and embedded into a LSA space via SVD
- finally, we train an XGBoost model on these LSA features


Implementation:

- I use the [ds-toolbox](https://github.com/alexeygrigorev/ds-toolbox) library for tokenization, vectorization, SVD, train/test split, etc
- The code originally comes from the [Mastering Java for Data Science](https://www.packtpub.com/big-data-and-business-intelligence/mastering-java-data-science) book (which, coincidentally, I authored)
- And actually the book might be quite helpful for this challenge
- Note that the library is quite raw, so if you notice a bug - feel free to create an issue. Pull requests are also welcome


## Running it:

For running it you need 

- Java 8
- Maven 3 
- [`ds-toolbox`](https://github.com/alexeygrigorev/ds-toolbox) and [`xboost4j`](https://github.com/dmlc/xgboost) (both not available on Maven Central)

Short instruction:

	# installing ds-toolbox

	git clone https://github.com/alexeygrigorev/ds-toolbox
	cd ds-toolbox/
	mvn install

	cd ..


	# installing xgboost and xgboost4j

	git clone --recursive https://github.com/dmlc/xgboost
	cd xgboost; make -j4

	cd jvm-packages/
	mvn -DskipTests install

	cd ../..


	# finally, running the benchmark code

	git clone https://github.com/alexeygrigorev/boosters-evotor-gettingstarted.git
	cd boosters-evotor-gettingstarted/
	mvn exec:java -Dexec.mainClass="evotor.GettingStarted"

