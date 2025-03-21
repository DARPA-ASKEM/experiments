# Reference: https://github.com/emmanuelle-dankwa/HAV-outbreak-Louisville/blob/main/surveillance/surveillance_main.Rmd

setwd("/Users/nliu/Workspace/projects/uncharted/askem/experiments/notebooks-r/data/hepatitis/repo")

# Load functions 
source("./surveillance_functions.R")     # All plot functions are written in this file

# Load data 

# Data on general population
# Full
dat2 <- readRDS(
    file = "./surveillance_data/dat2.RDS"
)
write.csv(dat2, "surveillance_data/dat2.csv", row.names = FALSE)

# Summarized (Weekly case counts in the general population)
dat2_yrwk <- readRDS(
    file = "./surveillance_data/dat2_yrwk.RDS"
)  
write.csv(dat2_yrwk, "./surveillance_data/dat2_yrwk.csv", row.names = FALSE)

# Weekly case count among PEH/PWUD (=risk group)
# Full
target <- readRDS(
    file = "./surveillance_data/target.RDS"
)
write.csv(target, "./surveillance_data/target.csv", row.names = FALSE)

# Summarized (Vaccination counts by week among PEH/PWUD)
target_yrwk <- readRDS(
    file = "./surveillance_data/target_yrwk.RDS"
)
write.csv(target_yrwk, "./surveillance_data/target_ywk.csv", row.names = FALSE)

# Vaccination data
# Summarized (Vaccination counts by week among PEH/PWUD)
vacc_counts_target <-  readRDS(
    file = "./surveillance_data/vacc_counts_target.RDS"
)
write.csv(vacc_counts_target, "./surveillance_data/vacc_counts_target.csv", row.names = FALSE)



