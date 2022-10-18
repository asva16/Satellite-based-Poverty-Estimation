library(readxl)
library(ggplot2)
library(reshape2)
library(ggtext)
library(plotly)
Augment <- read_excel("augment_Model.xlsx")
head(Augment)
colnames(Augment)[1]='Epoch'
Augment$model = 'Augment'

Base <- read_excel("Base_Model.xlsx")
colnames(Base)[1]='Epoch'
Base$model = 'Base'

BatchNorm <- read_excel("BatchNorm_Model.xlsx")
colnames(BatchNorm)[1]='Epoch'
BatchNorm$model = 'BatchNorm'

Dropout <- read_excel("Dropout_Model.xlsx")
colnames(Dropout)[1]='Epoch'
Dropout$model = 'Dropout'

L1 <- read_excel("L1_Model.xlsx")
colnames(L1)[1]='Epoch'
L1$model = 'L1'

L1L2 <- read_excel("L1L2_Model.xlsx")
colnames(L1L2)[1]='Epoch'
L1L2$model = 'L1L2'

L2 <- read_excel("L2_Model.xlsx")
colnames(L2)[1]='Epoch'
L2$model = 'L2'

all_model = rbind(Augment, Base, BatchNorm, Dropout, L1, L1L2, L2)
all_model$model = as.factor(all_model$model)
str(all_model)

ggplot(all_model) +
  aes(x = Epoch, y = loss, colour = model) +
  geom_line(size = 0.5) +
  geom_line(size=0.5, aes(x = Epoch, y = val_loss, colour = model)) +
  geom_point(aes(x = Epoch, y = val_loss, colour = model)) +
  scale_color_hue(direction = 1) +
  theme_minimal() + ylim(0,2)

ggplot(all_model) +
  aes(x = Epoch, y = loss) +
  geom_line(size = 0.5, colour = "red") +
  geom_line(aes(x = Epoch, y = val_loss), col='blue') +
  theme_minimal() +
  #facet_wrap(vars(model), scales = 'free_y') +
  labs(title = "<span style = 'color: red;'>Training Loss</span> vs <span style = 'color: blue;'>Validation Loss") +
  theme(plot.title = element_markdown()) +
  facet_trelliscope(
    ~ model,
    scales = 'free',
    ncol = 3, 
    nrow = 3,
    as_plotly = T)

ggplot(all_model) +
  aes(x = Epoch, y = accuracy) +
  geom_line(size = 0.5, colour = "red") +
  geom_line(aes(x = Epoch, y = val_accuracy), col='blue') +
  theme_minimal() +
  #facet_wrap(vars(model), scales = 'free_y') +
  labs(title = "<span style = 'color: red;'>Training Acc</span> vs <span style = 'color: blue;'>Validation Acc") +
  theme(plot.title = element_markdown()) +
  facet_trelliscope(
    ~ model,
    scales = 'free',
    ncol = 3, 
    nrow = 3,
    as_plotly = T)

