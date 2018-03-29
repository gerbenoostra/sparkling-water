.. _productionizing-h2o:

Productionizing H2O
===================

.. _about-pojo-mojo:

About POJOs and MOJOs
---------------------

H2O allows you to convert the models you have built to either a `Plain Old Java Object <https://en.wikipedia.org/wiki/Plain_Old_Java_Object>`__ (POJO) or a Model ObJect, Optimized (MOJO). 

H2O-generated MOJO and POJO models are intended to be easily embeddable in any Java environment. The only compilation and runtime dependency for a generated model is the ``h2o-genmodel.jar`` file produced as the build output of these packages. This file is a library that supports scoring. For POJOs, it contains the base classes from which the POJO is derived from. (You can see "extends GenModel" in a POJO class. The GenModel class is part of this library.) For MOJOs, it also contains the required readers and interpreters. The ``h2o-genmodel.jar`` file is required when POJO/MOJO models are deployed to production.

Users can refer to the Quick Start topics that follow for more information about generating POJOs and MOJOs.

Developers can refer to the the `POJO and MOJO Model Javadoc <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html>`__.


MOJO Quick Start
~~~~~~~~~~~~~~~~

This section describes how to build and implement a MOJO (Model Object, Optimized) to use predictive scoring. Java developers should refer to the `Javadoc <http://docs.h2o.ai/h2o/latest-stable/h2o-genmodel/javadoc/index.html>`__ for more information, including packages.

What is a MOJO?
'''''''''''''''

A MOJO (Model Object, Optimized) is an alternative to H2O's POJO. As with POJOs, H2O allows you to convert models that you build to MOJOs, which can then be deployed for scoring in real time.

**Note**: MOJOs are supported for Deep Learning, DRF, GBM, GLM, GLRM, K-Means, Stacked Ensembles, SVM, Word2vec, and XGBoost models.

Benefit over POJOs
''''''''''''''''''

While POJOs continue to be supported, some customers encountered issues with large POJOs not compiling. (Note that POJOs are not supported for source files larger than 1G.) MOJOs do not have a size restriction and address the size issue by taking the tree out of the POJO and using generic tree-walker code to navigate the model. The resulting executable is much smaller and faster than a POJO.

At large scale, new models are roughly 20-25 times smaller in disk space, 2-3 times faster during "hot" scoring (after JVM is able to optimize the typical execution paths), and 10-40 times faster in "cold" scoring (when JVM doesn't know yet know the execution paths) compared to POJOs. The efficiency gains are larger the bigger the size of the model.

H2O conducted in-house testing using models with 5000 trees of depth 25. At very small scale (50 trees / 5 depth), POJOs were found to perform ≈10% faster than MOJOs for binomial and regression models, but 50% slower than MOJOs for multinomial models.

Building a MOJO
'''''''''''''''
 When using scala with sparkling-water, a slightly different approach has to be taken as sparkling water does not generate a zip file.
 This approach will write the bytestream to a file, which then can be read back from an application without sparkling water.

 **Step 1: Build and extract a model **:

 1. Open a terminal window and start sparkling-water.
 2. Run the following commands to build a simple GBM model.

   .. code:: scala

       // Prepare the environment
       import org.apache.spark.h2o._
       import hex.genmodel.utils.DistributionFamily
       import hex.tree.gbm.GBMModel.GBMParameters
       import hex.tree.gbm.GBM
       val hc = H2OContext.getOrCreate(spark)

       // Load and preapre the data
       val table: H2OFrame = new H2OFrame(new java.io.File("examples/smalldata/prostate/prostate.csv"))
       val target = "CAPSULE"
       table.replace(table.find(target), table.vec(target).toCategoricalVec).remove()

       // Build GBM model
       val gbmParams = new GBMParameters()
       gbmParams._train = table._key
       gbmParams._response_column = target
       gbmParams._ntrees = 5
       gbmParams._nfolds = 3
       gbmParams._min_rows = 10
       gbmParams._distribution = DistributionFamily.multinomial
       val gbm = new GBM(gbmParams)
       val model = gbm.trainModel.get

        // Export the mojo
       val outputStream = new FileOutputStream(new File("model.mojo"))
       try {
         gbmModel.getMojo.writeTo(outputStream)
       }
       finally if (outputStream != null) outputStream.close()

 **Step 2: Compile and run the MOJO**
 1. Create a new file called main.scala. This file will be able to load the previously generated model.

   .. code:: scala

       import java.io.{File, FileInputStream}

       import hex.genmodel.easy.{EasyPredictModelWrapper, RowData}
       import hex.genmodel.{ModelMojoReader, MojoReaderBackendFactory}


       object Main extends App {

         override def main(args: Array[String]): Unit = {
           // Load the MOJO
           val is = new FileInputStream(new File("model.mojo"))
           val reader = MojoReaderBackendFactory.createReaderBackend(is, MojoReaderBackendFactory.CachingStrategy.MEMORY)
           val mojoModel = ModelMojoReader.readFrom(reader)

           // Setup predictor
           val config = new EasyPredictModelWrapper.Config()
           config.setModel(mojoModel)
           config.setConvertUnknownCategoricalLevelsToNa(true)
           val model = new EasyPredictModelWrapper(config)

           // Score a new sample
           val row = new RowData
           row.put("AGE", "68")
           row.put("RACE", "2")
           row.put("DCAPS", "2")
           row.put("VOL", "0")
           row.put("GLEASON", "6")
           val p = model.predictBinomial(row)

           println("Has penetrated the prostatic capsule (1=yes; 0=no): " + p.label)
           println("Class probabilities: " + p.classProbabilities.mkString(", "))
         }
       }


