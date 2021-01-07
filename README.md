# Moving Least Squares algorithm from scratch

To get better feeling of NumPy library with a bit of linear algebra, I tried to implement Moving Least Squares (MLS) algorithm. I tried to implement the algorith following this
paper: https://www.researchgate.net/publication/276238932_Measurement_Data_Fitting_Based_on_Moving_Least_Squares_Method. 

Based on my research, MLS algorithm is not well documented - I didn't find any useful coded solution on the internet, so I had to translate matemathical formulas from papers
into code. In the end, this personal project became quite hard to get working properly and I think there is still so many things to improve, but I became much more 
comfortable with NumPy and also revisited some linear algebra. After all, the main goal was to deepen my programming skills, so I am satisfied with the result.

## Content:
- There is code dealing with problem 1 in .py script, along with solution in Jupyter notebook.
- Details are described in report.pdf - this file contains commented code, along with bits of underlying theory used
- The file problem2.ipynb contains solution in Jupyter notebook of 2D problem, where the hyperplane is fitted to measured points

## Why MLS ?
Long story short - MLS algorithm can produce a good fit to the data using just low orders of polynomial basis. For example, in the picture below, we can get "closer" fit 
using only 2nd order polynomial, with comparison with Ordinary Least Squares algorithm (OLS) using 8th order polynomial. The main idea is making approximation based just on close surroundings of
particular points (weighting function is used for such task), not across the whole dataset.

## Conclusion
The resource article describes approximation of 1D and 2D problem. I implemented the algorithm using for loops and so this solution is not very convenient. If using large amounts 
of data, or even just using more than one independet variable, the execution time of the fitting procedure grows exponentialy. I guess there could be more convenient solution using vectorization, but I was not able to implement that 
into working code. This would be the main suggestion how to improve the code in the future.


![alt text](https://github.com/viliam-gago/moving_least_squares/blob/master/img/comparison.png?raw=true)
