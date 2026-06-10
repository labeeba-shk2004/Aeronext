# AeroNexus AI - Interview Preparation Guide

* **Technologies:** Python, Streamlit, Surprise Library (Collaborative Filtering / SVD / Matrix Factorization), Reinforcement Learning (Q-Learning), TextBlob NLP, Pandas, NumPy, Plotly

---

## 📝 Project Description
**AeroNexus AI** is an AI-driven Airport Revenue Optimization Platform designed to maximize non-aeronautical revenue (retail, Food & Beverage, and lounges) while improving passenger satisfaction. Deployed as an interactive Streamlit dashboard, the system simulates real-time passenger flows and commercial transactions for two different airport tiers:

* **Delhi Indira Gandhi International Airport (DEL):** Tier-1 Gateway (70M+ annual passengers) with a 55% business / 45% leisure traveler split, focusing on high-tech IoT integrations and premium lounge/retail personalization.
* **Jaipur International Airport (JAI):** Tier-2 Regional Hub (6.8M+ annual passengers) with a 30% business / 70% leisure traveler split, focusing on cost-effective, tourism-oriented offerings (local handicrafts and regional food).

The platform integrates three distinct AI pipelines:
1. **Personalized Recommendations:** Employs collaborative filtering (via the Surprise library) to generate context-specific, real-time product/lounge offers for passengers to drive retail conversions.
2. **Dynamic Pricing:** Operates a reinforcement learning (Q-Learning) agent to adjust product prices continuously based on retail categories, hourly demand, and crowd density.
3. **Aspect-Based NLP Feedback Monitoring:** Parses mixed passenger reviews to isolate operational pain points and calculate real-time Net Promoter Scores (NPS).

---

## 🛠️ Technology Stack & Libraries Explained Simply

* **Streamlit:** Used to build the interactive web dashboard. It lets developers build a clean user interface with dropdowns, charts, and tabs using only Python, avoiding the need for front-end code (HTML, CSS, JavaScript).
* **Surprise Library:** A specialized Python library for building recommender systems. It performs collaborative filtering algorithms (like matrix factorization) under the hood to identify passenger preferences.
* **TextBlob:** A simple, beginner-friendly Natural Language Processing (NLP) library used to parse review text and calculate a sentiment score (how positive, negative, or neutral a comment is).
* **Q-Learning (Reinforcement Learning):** An algorithmic approach where the model learns to make decisions (like adjusting prices) by testing actions in an environment and earning points (rewards) or losing points (penalties) based on performance.
* **Pandas:** Used for data manipulation. It organizes raw logs, transactions, and passenger lists into structured tables (called DataFrames) for easy filtering and querying.
* **NumPy:** Used for quick numerical operations, such as generating random dwell-time distributions or running mathematical squashing functions.
* **Plotly:** Used to generate interactive, hover-friendly charts and graphs (like NPS trends and peak hourly demand) for the dashboard.

---

## 📝 Bullet Points
* **End-to-End AI Platform:** Developed a Streamlit-based Airport Revenue Optimization platform comparing Tier-1 (DEL) and Tier-2 (JAI) hubs to maximize non-aeronautical revenue (retail, F&B, lounges).
* **Personalized Recommendations (Surprise):** Implemented a collaborative filtering model using the **Surprise library** (Matrix Factorization) to recommend real-time retail & F&B offers, driving a predicted **15% conversion lift**.
* **Dynamic Pricing (Q-Learning):** Engineered a tabular **Q-Learning** reinforcement learning agent that optimizes prices in real-time based on demand, time of day, and crowd levels, yielding a **12-18% revenue uplift**.
* **Aspect-Based Sentiment NLP:** Developed a sentiment tracking pipeline using **TextBlob** and domain-specific keywords to classify feedback into NPS buckets (Promoter, Passive, Detractor) and isolate issues across 6 operational aspects.

---

## 🔍 Core ML Models Explained

### 1. Collaborative Filtering (Surprise Library) 🛍️
* **How it works:** Deconstructs the sparse user-item interaction matrix into lower-dimensional user and item embeddings using matrix factorization. By calculating dot products of these latent vectors, the model predicts rating scores for items a passenger hasn't interacted with yet.
* **Cold Start Strategy:** If a user is new, it falls back to domain-specific popular products (e.g., Starbucks for F&B, local handicrafts for regional retail).
* **Business Metric:** Translates predicted rating confidence into conversion probabilities, predicting a **15% retail conversion lift**.

### 2. Dynamic Pricing Engine (Q-Learning RL) 📈
* **State Space:** 4-tuple `(demand_level, time_period, crowd_level, product_category)`.
* **Action Space:** Discrete price adjustments: `[-20%, -10%, 0%, +10%, +20%]`.
* **Policy:** Epsilon-Greedy (10% exploration of random actions, 90% exploitation of best known actions) to balance learning and earning.
* **Reward Function:** Scores pricing decisions by calculating the profit change, compressing it into a stable scale, and applying a heavy penalty if sales volume drops.
* **Safety Constraints:** Dynamic pricing limits price increases to a maximum of 10% during low-crowd periods, caps luxury items at +15%, and permits up to +20% for F&B during peak congestion.

### 3. Aspect-Based Sentiment & NPS Engine (NLP) 🗣️
* **Sentiment Polarity:** TextBlob sentiment polarity adjusted with domain-specific keyword scaling to give more weight to airport-related positive and negative words.
* **NPS Bucketing:** Classifies positive polarity (above 0.3) as Promoters, negative polarity (below -0.1) as Detractors, and the rest as Passives.
* **Aspect Analysis:** Splits feedback into sentences and targets 6 dimensions (`service`, `shopping`, `food`, `facility`, `navigation`, `wait_time`) using keyword lookup to help operations trace bottlenecks.

---

## 💡 Simple Explanation of the Models (With Examples)

### 1. Collaborative Filtering (Surprise Library) 🛍️
* **Simple Explanation:** Imagine a Netflix-style recommendation system for airport retail. The model creates user profiles by analyzing shopping behaviors. If Passenger A and Passenger B both buy travel pillows, but only Passenger A buys noise-canceling headphones, the model suggests headphones to Passenger B.
* **Example:** A business traveler at DEL T3 gate area receives a real-time coupon for 15% off at Starbucks because other business travelers with similar itineraries frequently purchase premium espresso before boarding.

### 2. Q-Learning for Dynamic Pricing 📈
* **Simple Explanation:** It behaves like a smart GPS that finds the fastest route based on real-time traffic. The agent checks crowd levels and times: if it's busy, it increases prices slightly (to maximize revenue); if it's quiet, it lowers prices or offers bundles (to clear inventory and drive volume).
* **Example:** At 2:00 PM (empty food court), the agent cuts the price of a Sandwich Combo by 10% to attract budget-conscious passengers. At 7:00 PM (evening peak flight boarding), the agent increases the price of that same combo by 15% because travelers are in a rush and demand is high.

### 3. Aspect-Based Sentiment & NPS Engine (NLP) 🗣️
* **Simple Explanation:** Rather than reading a review and giving a single score (like "Neutral"), the model divides reviews into separate sentences and tracks individual categories (Food, Wait Time, Service) so the management team knows exactly what to fix.
* **Example:** A passenger writes: *"The luxury shopping was amazing, but the gate seating area was filthy."* The engine splits this up: the `shopping` aspect gets a positive sentiment rating (e.g., +0.8), while the `facility` aspect gets a negative rating (e.g., -0.7), automatically alerting the janitorial team.

---

## 🎙️ Core Interview Q&A

### **Q1: Tell me about this project.**
> *"I built AeroNexus AI, an airport revenue optimization platform. It uses three ML models to boost non-aeronautical revenue: a Matrix Factorization recommendation engine (using the Surprise library) for retail offers, a Q-Learning agent for dynamic pricing, and an aspect-based NLP classifier to analyze passenger reviews. The project simulates operations at Delhi (DEL) and Jaipur (JAI) airports, achieving a targeted 25% revenue-per-passenger uplift."*

### **Q2: Why use the Surprise library / Collaborative Filtering instead of Deep Learning?**
> *"The Surprise library is optimized for Collaborative Filtering and Matrix Factorization algorithms. It is computationally lightweight and low-latency, which is essential for real-time coupon delivery at terminals. Deep Learning is resource-heavy, requires massive datasets, and introduces high inference latency, which doesn't scale well for real-time edge devices at airports."*

### **Q3: How does your Q-Learning agent determine pricing, and how do you prevent price gouging?**
> *"The agent maps the current environment (time, demand, category, crowd level) to a Q-value and selects a price adjustment between -20% and +20%. If a price hike causes sales volume to crash, the reward function applies a heavy negative penalty. We also enforce hard constraints—capping price hikes at +10% during low crowd periods and +15% for luxury items to protect the customer experience."*

### **Q4: How does your sentiment analyzer handle mixed reviews (e.g., 'Clean restroom, bad coffee')?**
> *"A basic sentiment model averages the sentence out as neutral. Our model splits reviews into sentences, uses keyword matching to map them to specific aspects ('clean restroom' $\rightarrow$ Facilities, 'bad coffee' $\rightarrow$ F&B), and scores them independently. This helps operators identify that restroom maintenance is performing well while food service needs attention."*

### **Q5: Why did you use the tanh function in the Reinforcement Learning agent's reward formula?**
> *"The reward function uses the hyperbolic tangent (tanh) mathematical function. It scales any arbitrary profit change into a stable, bounded range (between -1 and 1). Without this, extremely large transactions (like bulk duty-free luxury purchases) would generate massive rewards that throw off the learning process. Squashing this value ensures stable, gradual updates."*

### **Q6: How does the system handle and compare Tier-1 (Delhi) vs. Tier-2 (Jaipur) airports differently?**
> *"DEL and JAI have completely different demographic and spending profiles. DEL has 55% business travelers who prefer premium lounges and quick dining, whereas JAI has 70% leisure travelers interested in local souvenirs and traditional F&B. The system configures separate base spend rates (₹400 for DEL vs. ₹300 for JAI) and adjusts the model baselines. This side-by-side comparison allows airport management to decide whether to scale complex IoT integrations (best for DEL) or cost-effective regional offerings (best for JAI)."*

### **Q7: How did you evaluate or validate the recommendation engine's performance?**
> *"We evaluate the model's accuracy using RMSE (Root Mean Squared Error) on a train-test split of the transaction interaction logs, which is a standard metric in the Surprise library. Beyond predictive accuracy, we measure the business impact by calculating the expected conversion lift based on the model's average recommendation confidence, translating predictive gains directly into estimated retail revenue uplift."*
