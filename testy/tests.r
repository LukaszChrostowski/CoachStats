library(tinytest)

# load all functions
source("notebooks/dataCleaning.R")
# set working directory on fantastyczne_gole 
# setwd("~/Desktop/fantastyczne_gole")
# make sure that have all packages installed (from dataCleaning.R)

# generate random data
random_data <- as.data.frame(x = rnorm(100),
                             y = rnorm(100, mean = 1, sd = 3))

# unit tests
expect_silent(
  data <- get_shots(file_path = "data/events/events_England.json", name_detail = "EN")
)

expect_silent(
  data <- get_shots2(json_file = "data/la_liga_events/ (1).json")
)

expect_error(
  get_final_data(data = random_data)
)

data1 <- get_shots2(json_file = "data/la_liga_events/ (1).json")
data2 <- get_shots2(json_file = "data/la_liga_events/ (2).json")

expect_length(colnames(data1), 22)
expect_length(colnames(data2), 22)
expect_false(is.vector(data1))
expect_true(is.data.frame(data1))
