# 1. Libraries ----
rm(list=ls())

library("devtools")
library("FactoMineR")
library("factoextra")
library("reticulate")
library("ggplot")
library("dplyr")
library("reshape2")
devtools::install_github("teunbrand/elementalist")
library("elementalist")

# 2. Load data pickles ----
pd <- import("pandas")
# el_train <- pd$read_pickle("data/interim/elliptic_train.pkl")
ve_train <- pd$read_pickle("data/interim/vesta_train_imputed.pkl")
cat_vars <- pd$read_pickle("data/interim/categorical_features.pkl")
rm(pd)

# 3. Perform MCA only on selected variables ----
to_drop <- names(unlist(sapply(
  c("card", "DeviceInfo"), 
  function(y) grep(y,cat_vars)
)))
to_keep <- setdiff(cat_vars, to_drop)
rm(to_drop)

# 4.Levels will have to be processed ----
res <- lapply(ve_train[to_keep], function(x) length(levels(x)))
res[order(unlist(res), decreasing=TRUE)]
sum(unlist(res))

# 5. Keep only categorical data, to save memory ----
cat_df <- ve_train[to_keep]
rm(ve_train)
gc()

# 6. Perform MCA, Scree plot ----
res.mca <- MCA(cat_df, ncp=100, graph=FALSE)

fviz_screeplot(res.mca, addlabels = TRUE, ylim = c(0, 1))

# 7. Save transformed dataset, objects
py_save_object(
  res.mca$ind$coord, "data/interim/vesta_train_mca", pickle = "pickle"
)
py_save_object(
  res.mca, "models/vesta_train_mca", pickle = "pickle"
)

save(res.mca, file="vesta_mca.RData")
# save.image("vesta_mca_image.RData")


# Scree plot
head(res.mca$eig)
eig_df <- as.data.frame(res.mca) %>%
names(eig_df) <- c("eig", "Variance", "Cumulative Variance", "idx")
eig_df$idx <- c(1:dim(eig_df)[1])
eig_df_melt <- melt(eig_df, id.vars = c("eig", "idx"))

pdf(
  "reports/figures/vesta_mca.pdf", width=6, height=4, bg = "white"
)
ggplot(eig_df_melt, aes(x=idx, y=value, color=variable)) + 
  geom_line(size=1, key_glyph="point") + 
  # geom_point(aes(y=cum_var), colour="#009e73", size=1) +
  geom_line(size=1, key_glyph="point") +
  # geom_point(aes(y=var), colour="#0072b2", size=1) +
  theme(
    panel.background = element_rect(
      fill = "white", colour = "black", size=1, linetype="solid"
    ),
    panel.grid.major = element_line(
      size = 0.5, linetype = 'solid', colour = "grey"
    ),
    panel.grid.minor = element_line(
      size = 0.5, linetype = 'solid', colour = "grey"
    ),
    legend.position = c(0.7, 0.4),
    legend.key = element_rect(colour = NA, fill = NA),
    legend.key.height = unit(0.5, "cm"),
    legend.key.width = unit(1, "cm"),
    legend.background = element_rect_round(
      linetype=1, size=0.5, color="black", radius=unit(0.2, "cm")
    )
  ) + 
  xlab("Component") + ylab("Explained variance (cumulative)") +
  guides(
    color=guide_legend(override.aes = list(size = 3))
  ) + 
  labs(color="Variables") + 
  scale_color_manual(
    labels=c("Variance", "Cumulated variance"),
    values=c("#0072b2", "#009e73")
  )

dev.off() 


