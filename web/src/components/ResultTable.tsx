type Props = {
  columns?: string[];
  rows?: Array<Array<string | number | boolean | null | object>>;
};

export function ResultTable({ columns = [], rows = [] }: Props) {
  if (!columns.length || !rows.length) return null;

  return (
    <div className="result-table-shell">
      <div className="overflow-x-auto">
        <table className="result-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((value, colIndex) => (
                  <td key={`${rowIndex}-${colIndex}`}>
                    {typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
