library(tidyverse)
library(tidymodels)
library(textrecipes)
library(themis)
library(parsnip)

##### Data 

ratings <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv')
details <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv')



##### Preprocessing

games_unique <- details %>%
  drop_na(primary, boardgamecategory, description) %>%
  arrange(primary, -yearpublished) %>%
  filter(!duplicated(primary)) %>% 
  transmute(name = primary, description = description, category = boardgamecategory)


# tokenizer to unnest the categories
label_tokenizer <- function(x) {
  x %>% 
    str_split(", ") %>% 
    map(str_remove_all, "[:punct:]") %>% 
    map(str_replace_all, "  ", " ") %>% 
    map(str_replace_all, " ", "_")
}

games_categories_rec <- recipe(~ ., data = games_unique) %>% 
  step_tokenize(category, custom_token = label_tokenizer) %>% 
  step_tf(category, prefix = "is")
prep <- prep(games_categories_rec)
games_clean <- bake(prep, new_data = NULL)



##### Resampling

# change mutate statement depending on which category should be predicted
games_split <- games_clean %>% 
  mutate(category = ifelse(is_category_Card_Game == 1, "cardgame", "other")) %>% 
  select(name, description, category) %>% 
  initial_split(.8, strata = category)

games_train <- training(games_split)
games_test <- testing(games_split)

games_folds <- vfold_cv(games_train, v = 6, strata = category)



##### Model workflow

blacklist_words <- c("player", "players", "game")

description_tokenizer <- function(x) {
  x %>% 
    str_remove_all("[^\\s]*&[^\\s]*") %>% 
    str_remove_all("[[:punct:][:digit:][\\+=]]") %>% # remove punctuation, digits, + and =
    str_squish() %>% 
    str_to_lower() %>% 
    str_split(" ")
}

games_rec <- recipe(category ~ description, data = games_train) %>% 
  step_tokenize(description, custom_token = description_tokenizer) %>% 
  step_tokenfilter(description, max_tokens = 400) %>% 
  step_stopwords(description) %>% 
  step_stopwords(description, custom_stopword_source = blacklist_words) %>% 
  step_tfidf(description) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_smote(category) # minority class upsampling based on knn

svm_spec <- svm_linear() %>% 
  set_mode("classification") %>% 
  set_engine("LiblineaR")

games_wf <- workflow() %>% 
  add_recipe(games_rec) %>% 
  add_model(svm_spec)



##### Estimation/training/fitting

svm_res <- fit_resamples(
  games_wf,
  games_folds, 
  metrics = metric_set(accuracy, precision),
  control = control_resamples(save_pred = TRUE)
)

final_fit <- last_fit(games_wf, games_split, metrics = metric_set(accuracy, precision))

games_fit <- extract_fit_parsnip(final_fit$.workflow[[1]])



##### Visualization

tidy(games_fit) %>% 
  filter(term != "Bias") %>% 
  mutate(sign = ifelse(estimate > 0, "card games", "other games"),
         term = str_remove(term, "tfidf_description_"),
         val = abs(estimate)) %>% 
  group_by(sign) %>% 
  slice_max(val, n=15) %>% 
  ggplot(aes(val, fct_reorder(term, val), fill=sign)) + 
  geom_col(width = 0.3, show.legend = F) +
  geom_point(aes(color=sign), size = 4, show.legend = F) +
  facet_wrap(~sign, scales = "free") +
  labs(title = "Boardgame Genre Word Associations", subtitle = "Words which predict...",
       x = "Coefficient", y = NULL)

