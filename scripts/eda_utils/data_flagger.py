import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFlagger:
    def __init__(self):
        logging.info("DataFlagger initialized.")
    
    def flag_rows(self, df, index_list, flag_column):
        """
        Adds or updates a boolean flag column in the DataFrame.
        
        Args:
            df (pd.DataFrame): Original DataFrame.
            index_list (pd.Index or list): Indexes to flag.
            flag_column (str): Name of the column to set as flag.
        
        Returns:
            pd.DataFrame: DataFrame with flag column updated.
        """
        try:
            df[flag_column] = False  # Initialize
            df.loc[index_list, flag_column] = True
            logging.info(f"Flagged {len(index_list)} rows in column '{flag_column}'.")
        except Exception as e:
            logging.error(f"Error while flagging '{flag_column}': {e}")
        return df
