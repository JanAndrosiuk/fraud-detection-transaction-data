rm(list=ls())

library("FactoMineR")
library("factoextra")
library("reticulate")
pd <- import("pandas")
# el_train <- pd$read_pickle("data/interim/elliptic_train.pkl")
ve_train <- pd$read_pickle("data/interim/vesta_train_imputed.pkl")
cat_vars <- pd$read_pickle("data/interim/categorical_features.pkl")
rm(pd)

# Perform MCA only on selected variables
to_drop <- names(unlist(sapply(
  c("card", "DeviceInfo"), 
  function(y) grep(y,cat_vars)
)))
to_keep <- setdiff(cat_vars, to_drop)
rm(to_drop)

# Calculate how many levels will have to be processed in total
res <- lapply(ve_train[to_keep], function(x) length(levels(x)))
res[order(unlist(res), decreasing=TRUE)]
sum(unlist(res))

cat_df <- ve_train[to_keep]
rm(ve_train)
gc()

res.mca <- MCA(cat_df, ncp=100, graph=FALSE)
fviz_screeplot(res.mca, addlabels = TRUE, ylim = c(0, 1))

# Save transformed dataset to python pickle
py_save_object(
  res.mca$ind$coord, "data/interim/vesta_train_mca", pickle = "pickle"
)
py_save_object(
  res.mca, "models/vesta_train_mca", pickle = "pickle"
)

# Save objects for future R session
save(res.mca, file="vesta_mca.RData")
# save.image("vesta_mca_image.RData")