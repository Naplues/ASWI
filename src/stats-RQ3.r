library(effsize)
warnings()

calc_p_value <- function(U, P) {
  p_value <- c()
  # P
  p_value <- c(p_value, wilcox.test(U[, 2], P[, 2], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(U[, 2], P[, 3], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(U[, 2], P[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(U[, 2], P[, 5], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(U[, 2], P[, 6], paired = TRUE)$p.value)
  p_value
}

calc_cliff <- function(U, P, R, F) {
  cliff <- c()

  cliff <- c(cliff, cliff.delta(U[, 2], P[, 2])$estimate)
  cliff <- c(cliff, cliff.delta(U[, 2], P[, 3])$estimate)
  cliff <- c(cliff, cliff.delta(U[, 2], P[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(U[, 2], P[, 5])$estimate)
  cliff <- c(cliff, cliff.delta(U[, 2], P[, 6])$estimate)
  cliff
}

calc_power <- function(U, P, R, F) {
  power <- c()
  # I vs. II
  power <- c(power, cliff.delta(U[, 2], P[, 2])$magnitude)
  power <- c(power, cliff.delta(U[, 2], P[, 3])$magnitude)
  power <- c(power, cliff.delta(U[, 2], P[, 4])$magnitude)
  power <- c(power, cliff.delta(U[, 2], P[, 5])$magnitude)
  power <- c(power, cliff.delta(U[, 2], P[, 6])$magnitude)
  power
}

# TMI_LR, TMI_SVM, TMI_MNB, TMI_DT,
name <- c('LR', 'RF', 'DT', 'Boost', 'SVM')

U_data <- read.csv('C:/Users/gzq10/Desktop/Git/SAWI/analysis/ANTERIOR.csv', header = TRUE)
P_data <- read.csv('C:/Users/gzq10/Desktop/Git/SAWI/analysis/Supervised-B+C_p.csv', header = TRUE)
R_data <- read.csv('C:/Users/gzq10/Desktop/Git/SAWI/analysis/Supervised-B+C_r.csv', header = TRUE)
F_data <- read.csv('C:/Users/gzq10/Desktop/Git/SAWI/analysis/Supervised-B+C_f.csv', header = TRUE)


# ====================================== Calculate p values =====================================
data_p <- calc_p_value(U_data, P_data)
data_r <- calc_p_value(U_data, R_data)
data_f <- calc_p_value(U_data, F_data)
result <- data.frame(name, data_p, data_r, data_f)
write.table(result, '../analysis/RQ3-p-value.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate cliff values =====================================
data_p <- calc_cliff(U_data, P_data)
data_r <- calc_cliff(U_data, R_data)
data_f <- calc_cliff(U_data, F_data)
result <- data.frame(name, data_p, data_r, data_f)
write.table(result, '../analysis/RQ3-cliff.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate power values =====================================
data_p <- calc_power(U_data, P_data)
data_r <- calc_power(U_data, R_data)
data_f <- calc_power(U_data, F_data)
result <- data.frame(name, data_p, data_r, data_f)
write.table(result, '../analysis/RQ3-power.csv', row.names = FALSE, sep = ',')
