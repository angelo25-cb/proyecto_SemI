import numpy as np
import pandas as pd

class ExcessManager:
    """Gestión de desperdicio y excedentes"""
    
    def __init__(self, unit_cost: float = 10.0, selling_price: float = 25.0):
        self.unit_cost = unit_cost
        self.selling_price = selling_price
    
    def calculate_excess(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calcular desperdicio por sobreproducción"""
        # Si se produce según predicción, el exceso es producción - demanda real
        excess_units = np.maximum(0, predicted - actual)
        
        # Si se produce según predicción, el déficit es demanda real - producción
        deficit_units = np.maximum(0, actual - predicted)
        
        # Costo del desperdicio
        waste_cost = np.sum(excess_units) * self.unit_cost
        
        # Pérdida por venta no realizada (costo de oportunidad)
        lost_sales_cost = np.sum(deficit_units) * (self.selling_price - self.unit_cost)
        
        # Comparación con método baseline (producción = media histórica)
        baseline_production = np.mean(actual)
        baseline_excess = np.maximum(0, baseline_production - actual)
        baseline_waste_cost = np.sum(baseline_excess) * self.unit_cost
        
        # Reducción de desperdicio
        waste_reduction = ((baseline_waste_cost - waste_cost) / baseline_waste_cost) * 100
        
        return {
            'excess_units': np.sum(excess_units),
            'deficit_units': np.sum(deficit_units),
            'waste_cost': waste_cost,
            'lost_sales_cost': lost_sales_cost,
            'total_opportunity_cost': waste_cost + lost_sales_cost,
            'baseline_waste_cost': baseline_waste_cost,
            'waste_reduction_percent': waste_reduction,
            'economic_savings': baseline_waste_cost - waste_cost
        }
    
    def calculate_valorization(self, excess_units: float, valorization_rate: float = 0.3) -> Dict:
        """Calcular impacto de valorización de productos no vendidos"""
        # Porcentaje de productos recuperados mediante donación, reutilización, etc.
        valorized_units = excess_units * valorization_rate
        
        # Valor recuperado
        recovered_value = valorized_units * (self.unit_cost * 0.5)  # 50% del costo recuperado
        
        # Reducción neta de desperdicio
        net_waste_units = excess_units - valorized_units
        
        return {
            'valorization_rate': valorization_rate,
            'valorized_units': valorized_units,
            'recovered_value': recovered_value,
            'net_waste_units': net_waste_units,
            'waste_reduction_after_valorization': (valorized_units / excess_units) * 100
        }
    
    def generate_report(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generar reporte completo de impacto operativo y económico"""
        report = []
        
        for idx, row in results_df.iterrows():
            # Calcular excedentes para este modelo
            excess_data = self.calculate_excess(
                row.get('y_test', []), 
                row.get('y_pred', [])
            )
            
            # Calcular valorización
            valorization = self.calculate_valorization(excess_data['excess_units'])
            
            report.append({
                'Modelo': row.get('Model', 'Unknown'),
                'MAE': row.get('MAE', 0),
                'MAPE (%)': row.get('MAPE', 0),
                'Unidades_Desperdiciadas': excess_data['excess_units'],
                'Costo_Desperdicio': f"${excess_data['waste_cost']:,.2f}",
                'Ahorro_Economico': f"${excess_data['economic_savings']:,.2f}",
                'Reduccion_Desperdicio (%)': f"{excess_data['waste_reduction_percent']:.1f}%",
                'Unidades_Valorizadas': valorization['valorized_units'],
                'Valor_Recuperado': f"${valorization['recovered_value']:,.2f}",
                'Reduccion_Neta_Desperdicio (%)': f"{valorization['waste_reduction_after_valorization']:.1f}%"
            })
        
        return pd.DataFrame(report)