library(effsize)
warnings()

calc_p_value <- function(S, U2, U3) {
  p_value <- c()
  # II
  p_value <- c(p_value, wilcox.test(S[, 3], U2[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 6], U2[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 9], U2[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 12], U2[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 15], U2[, 4], paired = TRUE)$p.value)

  p_value <- c(p_value, wilcox.test(S[, 4], U3[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 7], U3[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 10], U3[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 13], U3[, 4], paired = TRUE)$p.value)
  p_value <- c(p_value, wilcox.test(S[, 16], U3[, 4], paired = TRUE)$p.value)

  p_value
}

calc_cliff <- function(S, U2, U3) {
  cliff <- c()
  # II
  cliff <- c(cliff, cliff.delta(S[, 3], U2[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 6], U2[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 9], U2[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 12], U2[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 15], U2[, 4])$estimate)

  cliff <- c(cliff, cliff.delta(S[, 4], U3[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 7], U3[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 10], U3[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 13], U3[, 4])$estimate)
  cliff <- c(cliff, cliff.delta(S[, 16], U3[, 4])$estimate)

  cliff
}

calc_power <- function(S, U2, U3) {
  power <- c()
  # I vs. II
  power <- c(power, cliff.delta(S[, 3], U2[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 6], U2[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 9], U2[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 12], U2[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 15], U2[, 4])$magnitude)

  power <- c(power, cliff.delta(S[, 4], U3[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 7], U3[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 10], U3[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 13], U3[, 4])$magnitude)
  power <- c(power, cliff.delta(S[, 16], U3[, 4])$magnitude)
  power
}

# TMI_LR, TMI_SVM, TMI_MNB, TMI_DT,
name <- c('LR', 'RF', 'DT', 'Boost', 'SVM', 'LR', 'RF', 'DT', 'Boost', 'SVM')

S_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ2-supervised_f.csv', header = TRUE)
U_II_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ4-unsupervised-II-f.csv', header = TRUE)
U_III_data <- read.csv('C:/Users/gzq-712/Desktop/Git/SAWI/analysis/RQ4-unsupervised-III-f.csv', header = TRUE)

# ====================================== Calculate p values =====================================
data <- calc_p_value(S_data, U_II_data, U_III_data)

result <- data.frame(name, data)
write.table(result, '../analysis/RQ4-p-value.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate cliff values =====================================
data <- calc_cliff(S_data, U_II_data, U_III_data)


result <- data.frame(name, data)
write.table(result, '../analysis/RQ4-cliff.csv', row.names = FALSE, sep = ',')


# ====================================== Calculate power values =====================================
data <- calc_power(S_data, U_II_data, U_III_data)

result <- data.frame(name, data)
write.table(result, '../analysis/RQ4-power.csv', row.names = FALSE, sep = ',')
