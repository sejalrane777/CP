import unittest
import pandas as pd
import numpy as np
from feature import embarkImputer, Mapper, age_col_tfr

class TestEmbarkImputer(unittest.TestCase):

    def test_impute_mode(self):
        data = {'Embarked': ['S', 'C', 'Q', np.nan, 'S']}
        df = pd.DataFrame(data)
        imputer = embarkImputer(variables='Embarked')
        imputer.fit(df)
        result = imputer.transform(df)
        expected_mode = df['Embarked'].mode()[0]
        self.assertEqual(result['Embarked'].iloc[3], expected_mode)

    def test_invalid_variable_type(self):
        with self.assertRaises(ValueError):
            embarkImputer(variables=123)

class TestMapper(unittest.TestCase):

    def test_mapping(self):
        data = {'Embarked': ['S', 'C', 'Q']}
        df = pd.DataFrame(data)
        mappings = {'S': 1, 'C': 2, 'Q': 3}
        mapper = Mapper(variables='Embarked', mappings=mappings)
        result = mapper.transform(df)
        expected = [1, 2, 3]
        self.assertListEqual(result['Embarked'].tolist(), expected)

    def test_invalid_variable_type(self):
        with self.assertRaises(ValueError):
            Mapper(variables=123, mappings={'S': 1})

class TestAgeColTfr(unittest.TestCase):

    def test_age_transformation(self):
        np.random.seed(42)
        data = {'Age': [22, 25, np.nan, 24, np.nan]}
        df = pd.DataFrame(data)
        age_transformer = age_col_tfr(variables='Age')
        age_transformer.fit(df)
        result = age_transformer.transform(df)
        self.assertFalse(result['Age'].isnull().any())
        self.assertTrue((result['Age'].iloc[2] >= (df['Age'].mean() - df['Age'].std())) and 
                        (result['Age'].iloc[2] <= (df['Age'].mean() + df['Age'].std())))

    def test_invalid_variable_type(self):
        with self.assertRaises(ValueError):
            age_col_tfr(variables=123)

if __name__ == '__main__':
    unittest.main()
