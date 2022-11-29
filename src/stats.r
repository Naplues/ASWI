library(effsize)
warnings()

calc_p_value <- function(data) {
  p_value <- c()
  # I vs. II
  p_value <- c(p_value, wilcox.test(data[, 2], data[, 3], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 5], data[, 6], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 8], data[, 9], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 11], data[, 12], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 14], data[, 15], paired = TRUE)$p.value)
  # I vs. III
  p_value <- c(p_value, wilcox.test(data[, 2], data[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 5], data[, 7], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 8], data[, 10], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 11], data[, 13], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(data[, 14], data[, 16], paired = TRUE)$p.value)

  p_value
}

calc_cliff <- function(data) {
  cliff <- c()
  # I vs. II
  cliff <- c(cliff, cliff.delta(data[, 2], data[, 3])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 5], data[, 6])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 8], data[, 9])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 11], data[, 12])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 14], data[, 15])$estimate)

  # I vs. III
  cliff <- c(cliff, cliff.delta(data[, 2], data[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 5], data[, 7])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 8], data[, 10])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 11], data[, 13])$estimate)
  cliff <- c(cliff, cliff.delta(data[, 14], data[, 16])$estimate)

  cliff
}

calc_power <- function(data) {
  power <- c()
  # I vs. II
  power <- c(power, cliff.delta(data[, 2], data[, 3])$magnitude)
  power <- c(power, cliff.delta(data[, 5], data[, 6])$magnitude)
  power <- c(power, cliff.delta(data[, 8], data[, 9])$magnitude)
  power <- c(power, cliff.delta(data[, 11], data[, 12])$magnitude)
  power <- c(power, cliff.delta(data[, 14], data[, 15])$magnitude)

  # I vs. III
  power <- c(power, cliff.delta(data[, 2], data[, 4])$magnitude)
  power <- c(power, cliff.delta(data[, 5], data[, 7])$magnitude)
  power <- c(power, cliff.delta(data[, 8], data[, 10])$magnitude)
  power <- c(power, cliff.delta(data[, 11], data[, 13])$magnitude)
  power <- c(power, cliff.delta(data[, 14], data[, 16])$magnitude)
  power
}

# TMI_LR, TMI_SVM, TMI_MNB, TMI_DT,
name <- c('LR', 'RF', 'DT', 'Boost', 'AUC', 'LR', 'RF', 'DT', 'Boost', 'AUC')

P_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ2-supervised_p.csv', header = TRUE)
R_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ2-supervised_r.csv', header = TRUE)
F_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ2-supervised_f.csv', header = TRUE)
A_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ2-supervised_a.csv', header = TRUE)

# ====================================== Calculate p values =====================================
Precision <- calc_p_value(P_data)
Recall <- calc_p_value(R_data)
F1 <- calc_p_value(F_data)
AUC <- calc_p_value(A_data)

result <- data.frame(name, Precision, Recall, F1, AUC)
write.table(result, '../analysis/RQ2-p-value.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate cliff values =====================================
Precision <- calc_cliff(P_data)
Recall <- calc_cliff(R_data)
F1 <- calc_cliff(F_data)
AUC <- calc_cliff(A_data)

result <- data.frame(name, Precision, Recall, F1, AUC)
write.table(result, '../analysis/RQ2-cliff.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate power values =====================================
Precision <- calc_power(P_data)
Recall <- calc_power(R_data)
F1 <- calc_power(F_data)
AUC <- calc_power(A_data)

result <- data.frame(name, Precision, Recall, F1, AUC)
write.table(result, '../analysis/RQ2-power.csv', row.names = FALSE, sep = ',')
