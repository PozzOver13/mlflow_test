# Tracking

[https://www.mlflow.org/docs/TRACKING](https://www.mlflow.org/docs/latest/tracking.html)

## Which are the core elements of MLflow Tracking?
We can synthetize the main component of tracking as follows:
1. **Experiments** = it is possible to create groups of "Runs" for specific tasks.
   1. **Runs** = executions of some piece of data science code
      1. Code version
      2. Start & End Time
      3. Source
      4. Parameters 
      5. Metrics
      6. Artifacts

## Where Runs are recorded?
By default, the MLflow Python API logs runs locally to files in a mlruns directory wherever you ran your program.
To log runs remotely, set the MLFLOW_TRACKING_URI environment variable.

4 remote tracking URIs (Uniform Resource Identifier):
1. Local file path (specified as file:/my/local/dir)
2. Database (supported dialects mysql, mssql, sqlite, and postgresql)
3. HTTP server 
4. Databricks workspace
