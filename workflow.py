"""
Data Analyst Agent - Conversational CSV Analysis Workflow

This workflow provides a conversational interface for analyzing CSV files from Google Drive.
It supports:
- Loading CSV files from Google Drive links
- Describing columns and data statistics
- Filtering rows based on conditions
- Calculating aggregates by group
- Maintaining conversation state (remembering the loaded file)
"""

import io
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from mistralai.workflows import (
    Activity,
    ActivityResult,
    Workflow,
    activity,
    workflow,
)
from mistralai.workflows.context import WorkflowContext

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

@dataclass
class ConversationState:
    """State maintained across conversation turns."""
    csv_data: Optional[pd.DataFrame] = None
    csv_file_name: Optional[str] = None
    csv_columns: Optional[list[str]] = None
    conversation_history: list[dict] = field(default_factory=list)

# Global state storage
_conversation_states: dict[str, ConversationState] = {}

def get_or_create_state(workflow_id: str) -> ConversationState:
    """Get or create conversation state for a workflow."""
    if workflow_id not in _conversation_states:
        _conversation_states[workflow_id] = ConversationState()
    return _conversation_states[workflow_id]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def parse_google_drive_link(link: str) -> Optional[str]:
    """Extract file ID from Google Drive link."""
    patterns = [
        r'/file/d/([^/]+)',
        r'id=([^&]+)',
        r'([a-zA-Z0-9_-]{33,})',
    ]
    for pattern in patterns:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    return None

def clean_column_name(name: str) -> str:
    """Clean column name for safer access."""
    return str(name).strip()# ============================================================================
# GOOGLE DRIVE INTEGRATION
# ============================================================================

@activity
async def fetch_google_drive_file(ctx: WorkflowContext, file_id: str) -> ActivityResult:
    """
    Fetch a file from Google Drive using the MCP connector.
    """
    try:
        if hasattr(ctx, 'call_tool'):
            result = await ctx.call_tool(
                tool_name='google_drive_mcp.download_file_content',
                arguments={'file_id': file_id}
            )
        else:
            try:
                from google_drive_mcp import download_file_content
                result = await download_file_content(file_id=file_id)
            except (ImportError, Exception):
                return ActivityResult(
                    success=False,
                    error="Google Drive MCP connector not connected. Please add and authenticate it in Mistral Studio.",
                    output=None
                )

        if not result or 'content' not in result:
            return ActivityResult(
                success=False,
                error=f"Could not download file with ID: {file_id}",
                output=None
            )
        return ActivityResult(success=True, output=result)
    except Exception as e:
        return ActivityResult(
            success=False,
            error=f"Error accessing Google Drive: {str(e)}",
            output=None
        )

# ============================================================================
# CORE ACTIVITIES (Fixed tools - no arbitrary code execution)
# ============================================================================

@activity
async def load_csv_from_google_drive(ctx: WorkflowContext, file_id: str, file_name: str) -> ActivityResult:
    """Load CSV file from Google Drive."""
    fetch_result = await fetch_google_drive_file(ctx, file_id)
    if not fetch_result.success:
        return fetch_result

    try:
        content = fetch_result.output.get('content')
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        elif not isinstance(content, str):
            content = str(content)

        df = pd.read_csv(io.StringIO(content))
        return ActivityResult(
            success=True,
            output={
                'dataframe': df,
                'file_name': file_name,
                'columns': list(df.columns),
                'shape': df.shape
            }
        )
    except Exception as e:
        return ActivityResult(
            success=False,
            error=f"Error parsing CSV: {str(e)}",
            output=None
        )

@activity
async def describe_dataframe(ctx: WorkflowContext, df: pd.DataFrame) -> ActivityResult:
    """Describe the DataFrame: columns, types, statistics."""
    try:
        description = {
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'shape': df.shape,
            'stats': df.describe(include='all').to_dict(),
            'sample': df.head(5).to_dict('records'),
            'missing_values': df.isnull().sum().to_dict(),
        }
        return ActivityResult(success=True, output=description)
    except Exception as e:
        return ActivityResult(
            success=False,
            error=f"Error describing data: {str(e)}",
            output=None
        )

@activity
async def filter_dataframe(
    ctx: WorkflowContext,
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any
) -> ActivityResult:
    """Filter DataFrame rows based on a condition."""
    try:
        column = clean_column_name(column)
        if column not in df.columns:
            return ActivityResult(
                success=False,
                error=f"Column '{column}' not found. Available: {list(df.columns)}",
                output=None
            )

        if operator == '==':
            filtered = df[df[column] == value]
        elif operator == '!=':
            filtered = df[df[column] != value]
        elif operator == '>':
            filtered = df[df[column] > value]
        elif operator == '<':
            filtered = df[df[column] < value]
        elif operator == '>=':
            filtered = df[df[column] >= value]
        elif operator == '<=':
            filtered = df[df[column] <= value]
        elif operator == 'contains':
            filtered = df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        elif operator == 'in':
            if not isinstance(value, list):
                value = [value]
            filtered = df[df[column].isin(value)]
        else:
            return ActivityResult(
                success=False,
                error=f"Unsupported operator: {operator}. Supported: ==, !=, >, <, >=, <=, contains, in",
                output=None
            )

        return ActivityResult(
            success=True,
            output={
                'filtered_data': filtered,
                'original_rows': len(df),
                'filtered_rows': len(filtered),
                'column': column,
                'operator': operator,
                'value': value
            }
        )
    except Exception as e:
        return ActivityResult(
            success=False,
            error=f"Error filtering data: {str(e)}",
            output=None
        )@activity
async def aggregate_dataframe(
    ctx: WorkflowContext,
    df: pd.DataFrame,
    group_by: list[str],
    aggregations: dict[str, list[str]]
) -> ActivityResult:
    """Calculate aggregates by group."""
    try:
        group_by = [clean_column_name(col) for col in group_by]
        for col in group_by:
            if col not in df.columns:
                return ActivityResult(
                    success=False,
                    error=f"Group column '{col}' not found",
                    output=None
                )
        for col in aggregations.keys():
            clean_col = clean_column_name(col)
            if clean_col not in df.columns:
                return ActivityResult(
                    success=False,
                    error=f"Aggregation column '{col}' not found",
                    output=None
                )

        agg_dict = {}
        for col, funcs in aggregations.items():
            clean_col = clean_column_name(col)
            agg_dict[clean_col] = []
            for func in funcs:
                func_lower = func.lower()
                if func_lower in ['sum', 'mean', 'min', 'max', 'count', 'std', 'var']:
                    agg_dict[clean_col].append(func_lower)
                else:
                    return ActivityResult(
                        success=False,
                        error=f"Unsupported aggregation function: {func}. Supported: sum, mean, min, max, count, std, var",
                        output=None
                    )

        result = df.groupby(group_by).agg(agg_dict)
        return ActivityResult(
            success=True,
            output={
                'aggregation_result': result,
                'group_columns': group_by,
                'aggregations': aggregations
            }
        )
    except Exception as e:
        return ActivityResult(
            success=False,
            error=f"Error aggregating data: {str(e)}",
            output=None
        )

# ============================================================================
# INTENT RECOGNITION
# ============================================================================

def extract_intent(user_message: str, state: ConversationState) -> dict:
    """Extract intent from user message."""
    message_lower = user_message.lower()

    # Intent: Load CSV from Google Drive
    if any(keyword in message_lower for keyword in ['charge', 'load', 'importe', 'import', 'ouvre', 'open', 'fichier', 'file', 'csv', 'google drive', 'drive.google.com']):
        urls = re.findall(r'https?://[^\s]+', user_message)
        if urls:
            link = urls[0]
            file_id = parse_google_drive_link(link)
            if file_id:
                file_name = link.split('/')[-1].split('?')[0]
                if file_name and '.' not in file_name:
                    file_name = f"{file_name}.csv"
                return {
                    'intent': 'load_csv',
                    'parameters': {
                        'link': link,
                        'file_id': file_id,
                        'file_name': file_name
                    }
                }
        return {
            'intent': 'load_csv',
            'parameters': {'error': 'No valid Google Drive link found.'}
        }

    # Intent: Describe data
    if any(keyword in message_lower for keyword in ['décris', 'describe', 'colonnes', 'columns', 'structure', 'statistiques', 'stats']):
        if state.csv_data is None:
            return {'intent': 'describe', 'parameters': {'error': 'No CSV file loaded yet.'}}
        return {'intent': 'describe', 'parameters': {}}

    # Intent: Filter data
    if any(keyword in message_lower for keyword in ['filtre', 'filter', 'montre', 'show', 'trouve', 'find', 'où', 'where']):
        if state.csv_data is None:
            return {'intent': 'filter', 'parameters': {'error': 'No CSV file loaded yet.'}}

        patterns = [
            r'([\w\s]+)\s*(>|<|>=|<=|==|!=|contains|in)\s*([\w\s,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                column = match.group(1).strip()
                operator = match.group(2).strip()
                value = match.group(3).strip()
                try:
                    if ',' in value:
                        value = [v.strip() for v in value.split(',')]
                    elif value.replace('.', '').replace('-', '').isdigit():
                        value = float(value) if '.' in value else int(value)
                except:
                    pass
                return {
                    'intent': 'filter',
                    'parameters': {
                        'column': column,
                        'operator': operator,
                        'value': value
                    }
                }
        return {
            'intent': 'filter',
            'parameters': {'error': 'Could not parse filter. Use: "filtre colonne > valeur"'}
        }

    # Intent: Aggregate data
    if any(keyword in message_lower for keyword in ['aggrège', 'aggregate', 'moyenne', 'mean', 'somme', 'sum', 'compte', 'count', 'par', 'by']):
        if state.csv_data is None:
            return {'intent': 'aggregate', 'parameters': {'error': 'No CSV file loaded yet.'}}

        by_match = re.search(r'par\s+([\w\s,]+)|by\s+([\w\s,]+)', message_lower, re.IGNORECASE)
        group_by = []
        if by_match:
            group_by = [col.strip() for col in (by_match.group(1) or by_match.group(2)).split(',')]

        agg_patterns = [
            (r'moyenne\s+(?:des\s+)?([\w\s]+)', 'mean'),
            (r'mean\s+of\s+([\w\s]+)', 'mean'),
            (r'somme\s+(?:des\s+)?([\w\s]+)', 'sum'),
            (r'sum\s+of\s+([\w\s]+)', 'sum'),
            (r'compte\s+(?:des\s+)?([\w\s]+)', 'count'),
            (r'count\s+of\s+([\w\s]+)', 'count'),
        ]
        aggregations = {}
        for pattern, func in agg_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                column = match.group(1).strip()
                if column not in aggregations:
                    aggregations[column] = []
                aggregations[column].append(func)

        if not group_by and state.csv_columns:
            for col in state.csv_columns:
                if col.lower() in ['region', 'category', 'ville', 'date']:
                    group_by.append(col)
                    break

        if not aggregations and state.csv_data is not None:
            numeric_cols = [col for col in state.csv_data.columns if pd.api.types.is_numeric_dtype(state.csv_data[col])]
            if numeric_cols:
                aggregations = {numeric_cols[0]: ['sum', 'mean', 'count']}

        if group_by and aggregations:
            return {
                'intent': 'aggregate',
                'parameters': {
                    'group_by': group_by,
                    'aggregations': aggregations
                }
            }
        return {
            'intent': 'aggregate',
            'parameters': {'error': 'Could not parse aggregation. Use: "moyenne de X par Y"'}
        }

    # Intent: Help
    if any(keyword in message_lower for keyword in ['aide', 'help', 'comment', 'how']):
        return {'intent': 'help', 'parameters': {}}

    # Intent: Reset
    if any(keyword in message_lower for keyword in ['efface', 'clear', 'réinitialise', 'reset']):
        return {'intent': 'reset', 'parameters': {}}

    # Default
    if state.csv_data is not None:
        return {'intent': 'query', 'parameters': {'question': user_message}}
    return {'intent': 'unknown', 'parameters': {'message': user_message}}# ============================================================================
# WORKFLOW
# ============================================================================

@workflow
class DataAnalystAgent:
    """Conversational data analysis agent workflow."""

    def __init__(self):
        self.workflow_id = None

    async def __call__(self, ctx: WorkflowContext, user_message: str) -> dict:
        """Main workflow entry point."""
        if not hasattr(self, 'workflow_id') or self.workflow_id is None:
            self.workflow_id = str(ctx.workflow_id) if ctx.workflow_id else "default"

        state = get_or_create_state(self.workflow_id)
        state.conversation_history.append({
            'role': 'user',
            'message': user_message,
            'timestamp': pd.Timestamp.now().isoformat()
        })

        intent_result = extract_intent(user_message, state)
        intent = intent_result.get('intent')
        parameters = intent_result.get('parameters', {})

        # Handle each intent
        if intent == 'load_csv':
            return await self._handle_load_csv(ctx, parameters, state)
        elif intent == 'describe':
            return await self._handle_describe(ctx, parameters, state)
        elif intent == 'filter':
            return await self._handle_filter(ctx, parameters, state)
        elif intent == 'aggregate':
            return await self._handle_aggregate(ctx, parameters, state)
        elif intent == 'help':
            return self._handle_help(state)
        elif intent == 'reset':
            return self._handle_reset(state)
        elif intent == 'query':
            return await self._handle_query(ctx, parameters, state)
        else:
            return self._handle_unknown(user_message, state)

    async def _handle_load_csv(self, ctx: WorkflowContext, parameters: dict, state: ConversationState) -> dict:
        """Handle CSV loading from Google Drive."""
        if 'error' in parameters:
            return {'response': f"❌ {parameters['error']}", 'state': {'csv_loaded': False}}

        link = parameters.get('link', '')
        file_id = parameters.get('file_id', '')
        file_name = parameters.get('file_name', 'data.csv')

        if not file_id:
            return {'response': "❌ Lien Google Drive invalide. Veuillez fournir un lien valide.", 'state': {'csv_loaded': False}}

        result = await load_csv_from_google_drive(ctx, file_id, file_name)
        if not result.success:
            return {'response': f"❌ Erreur: {result.error}", 'state': {'csv_loaded': False}}

        state.csv_data = result.output['dataframe']
        state.csv_file_name = result.output['file_name']
        state.csv_columns = result.output['columns']

        return {
            'response': f"✅ Fichier **{file_name}** chargé !\n\n"
                       f"📊 **{result.output['shape'][0]} lignes**, **{result.output['shape'][1]} colonnes**\n\n"
                       f"📝 Colonnes: {', '.join(result.output['columns'])}\n\n"
                       f"💬 Que voulez-vous analyser ? (ex: 'décris', 'filtre ventes > 1000', 'moyenne par région')",
            'state': {'csv_loaded': True, 'file_name': file_name, 'columns': result.output['columns']}
        }

    async def _handle_describe(self, ctx: WorkflowContext, parameters: dict, state: ConversationState) -> dict:
        """Handle data description request."""
        if state.csv_data is None:
            return {'response': "❌ Aucun fichier chargé. Chargez un fichier d'abord.", 'state': {'csv_loaded': False}}

        result = await describe_dataframe(ctx, state.csv_data)
        if not result.success:
            return {'response': f"❌ Erreur: {result.error}", 'state': {'csv_loaded': True}}

        description = result.output
        response = f"📊 **Description de {state.csv_file_name}**\n\n"
        response += f"📝 **Colonnes ({len(description['columns'])}):**\n"
        for col, dtype in description['dtypes'].items():
            response += f"  - `{col}`: {dtype}\n"

        response += f"\n📏 **Forme:** {description['shape'][0]} lignes × {description['shape'][1]} colonnes\n\n"
        response += "📈 **Statistiques:**\n"
        for col, stats in description['stats'].items():
            if isinstance(stats, dict) and 'mean' in stats:
                mean_val = stats.get('mean', 'N/A')
                if isinstance(mean_val, (int, float)):
                    mean_val = f"{mean_val:.2f}"
                response += f"  - `{col}`: min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, mean={mean_val}\n"

        missing = description['missing_values']
        if any(missing.values()):
            response += f"\n🔍 **Valeurs manquantes:**\n"
            for col, count in missing.items():
                if count > 0:
                    response += f"  - `{col}`: {count} valeurs\n"
        else:
            response += f"\n✅ Aucune valeur manquante\n"

        response += f"\n🎯 **Échantillon (3 lignes):**\n"
        for row in description['sample'][:3]:
            response += f"  {row}\n"

        return {'response': response, 'state': {'csv_loaded': True}}

    async def _handle_filter(self, ctx: WorkflowContext, parameters: dict, state: ConversationState) -> dict:
        """Handle data filtering request."""
        if state.csv_data is None:
            return {'response': "❌ Aucun fichier chargé.", 'state': {'csv_loaded': False}}
        if 'error' in parameters:
            return {'response': f"❌ {parameters['error']}", 'state': {'csv_loaded': True}}

        column = parameters.get('column', '')
        operator = parameters.get('operator', '')
        value = parameters.get('value', '')
        result = await filter_dataframe(ctx, state.csv_data, column, operator, value)

        if not result.success:
            return {'response': f"❌ Erreur: {result.error}", 'state': {'csv_loaded': True}}

        filter_info = result.output
        filtered_df = filter_info['filtered_data']
        response = f"🔍 **Filtrage:** `{column}` {operator} `{value}`\n\n"
        response += f"📊 **Résultats:** {filter_info['filtered_rows']} lignes (sur {filter_info['original_rows']})\n\n"

        if len(filtered_df) > 0:
            response += "📋 **Première ligne:**\n"
            first_row = filtered_df.iloc[0].to_dict()
            for col, val in first_row.items():
                response += f"  - {col}: {val}\n"
            if len(filtered_df) > 1:
                response += f"\n... et {len(filtered_df) - 1} autres lignes\n"
        else:
            response += "⚠️ Aucun résultat\n"

        response += "\n💡 **Prochaine étape:**\n  - Autre filtre ?\n  - Agrégation sur ce résultat ?\n"
        return {'response': response, 'state': {'csv_loaded': True, 'last_filter': filter_info}}

    async def _handle_aggregate(self, ctx: WorkflowContext, parameters: dict, state: ConversationState) -> dict:
        """Handle aggregation request."""
        if state.csv_data is None:
            return {'response': "❌ Aucun fichier chargé.", 'state': {'csv_loaded': False}}
        if 'error' in parameters:
            return {'response': f"❌ {parameters['error']}", 'state': {'csv_loaded': True}}

        group_by = parameters.get('group_by', [])
        aggregations = parameters.get('aggregations', {})

        if not group_by or not aggregations:
            return {'response': "❌ Impossible de comprendre l'agrégation. Essayez: 'moyenne des ventes par région'", 'state': {'csv_loaded': True}}

        result = await aggregate_dataframe(ctx, state.csv_data, group_by, aggregations)
        if not result.success:
            return {'response': f"❌ Erreur: {result.error}", 'state': {'csv_loaded': True}}

        agg_info = result.output
        agg_result = agg_info['aggregation_result']
        response = f"📊 **Agrégation par:** {', '.join(group_by)}\n\n📈 **Résultats:**\n"

        for i, (group_key, group_data) in enumerate(agg_result.iterrows()):
            if i >= 10:
                response += f"\n... et {len(agg_result) - 10} autres groupes\n"
                break
            group_str = str(group_key) if not isinstance(group_key, tuple) else " × ".join(str(k) for k in group_key)
            response += f"\n  **{group_str}:**\n"
            for col, values in group_data.items():
                if isinstance(values, pd.Series):
                    for func, val in values.items():
                        if pd.notna(val):
                            response += f"    - {col} ({func}): {val:.2f}\n"
                else:
                    response += f"    - {col}: {values:.2f}\n"

        return {'response': response, 'state': {'csv_loaded': True, 'last_aggregation': agg_info}}

    def _handle_help(self, state: ConversationState) -> dict:
        """Handle help request."""
        response = "🤖 **Aide - Agent Analyste de Données**\n\n"
        response += "**📥 Charger un fichier:**\n"
        response += "  - `Charge le fichier https://drive.google.com/file/d/...`\n\n"
        response += "**📊 Décrire les données:**\n"
        response += "  - `Décris les colonnes`\n"
        response += "  - `Quelles sont les statistiques ?`\n\n"
        response += "**🔍 Filtrer:**\n"
        response += "  - `Filtre les ventes > 1000`\n"
        response += "  - `Montre où age > 30`\n\n"
        response += "**📈 Agrégats:**\n"
        response += "  - `Moyenne des ventes par région`\n"
        response += "  - `Somme du CA par catégorie`\n"
        response += "  - `Compte les clients par ville`\n\n"
        response += "**🔄 Autres:**\n"
        response += "  - `Aide` - Cette aide\n"
        response += "  - `Réinitialise` - Effacer le fichier\n"

        if state.csv_data is not None:
            response += f"\n\n📌 **Fichier chargé:** {state.csv_file_name}\n"
            response += f"   Lignes: {len(state.csv_data)}, Colonnes: {len(state.csv_columns)}\n"
        else:
            response += "\n\n📌 **Aucun fichier chargé**\n"

        return {'response': response, 'state': {'csv_loaded': state.csv_data is not None}}

    def _handle_reset(self, state: ConversationState) -> dict:
        """Handle reset request."""
        state.csv_data = None
        state.csv_file_name = None
        state.csv_columns = None
        return {
            'response': "🔄 **État réinitialisé.**\n\nLe fichier a été effacé. Chargez un nouveau fichier avec:\n"
                       "`Charge le fichier https://drive.google.com/file/d/...`",
            'state': {'csv_loaded': False}
        }

    async def _handle_query(self, ctx: WorkflowContext, parameters: dict, state: ConversationState) -> dict:
        """Handle general query."""
        question = parameters.get('question', '')
        response = f"💭 **Votre question:** {question}\n\n"

        if state.csv_data is not None:
            response += f"📊 **Fichier:** {state.csv_file_name} ({len(state.csv_data)} lignes)\n\n"
            response += "💡 **Suggestions:**\n"
            response += "   - `Décris les données`\n"
            response += "   - `Filtre colonne > valeur`\n"
            response += "   - `Moyenne de X par Y`\n"
        else:
            response += "❌ Aucun fichier chargé. Utilisez: `Charge le fichier ...`\n"

        return {'response': response, 'state': {'csv_loaded': state.csv_data is not None}}

    def _handle_unknown(self, message: str, state: ConversationState) -> dict:
        """Handle unknown intent."""
        response = f"❓ Je n'ai pas compris: **{message}**\n\n"
        if state.csv_data is not None:
            response += "💡 **Essayez:**\n"
            response += "   - `Décris`\n"
            response += "   - `Filtre X > Y`\n"
            response += "   - `Moyenne par groupe`\n"
            response += "   - `Aide`\n"
        else:
            response += "📥 **Chargez un fichier:** `Charge le fichier https://drive.google.com/file/d/...`\n"
        return {'response': response, 'state': {'csv_loaded': state.csv_data is not None}}

if __name__ == "__main__":
    print("Data Analyst Agent Workflow - Ready for Mistral Cloud deployment")
