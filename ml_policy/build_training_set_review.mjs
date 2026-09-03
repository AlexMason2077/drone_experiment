import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const sourceCsv = "/Users/alexmason/Downloads/drone_experiment/analysis_outputs/ml_policy/expanded_25m/oracle_training_states_0p25_25m.csv";
const outputDir = "/Users/alexmason/Downloads/drone_experiment/outputs/training_set_review";
const outputXlsx = `${outputDir}/oracle_training_set_review_0p25_25m.xlsx`;
const previewDir = `${outputDir}/previews`;

const csvText = await fs.readFile(sourceCsv, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Raw Training Data" });
const raw = workbook.worksheets.getItem("Raw Training Data");
const rawValues = raw.getUsedRange(true).values;
const headers = rawValues[0];
const rows = rawValues.slice(1);
const headerIndex = new Map(headers.map((header, index) => [header, index]));

const selectedColumns = [
  "scenario_id",
  "wind_direction",
  "wind_level",
  "charging_pad_count",
  "remaining_distance_m",
  "history_steps",
  "history_distance_m",
  "soc_d1",
  "soc_d2",
  "soc_d3",
  "soc_d4",
  "soc_d5",
  "soc_range",
  "oracle_structure",
  "oracle_position_json",
  "oracle_total_minutes",
  "oracle_charging_minutes",
  "oracle_second_structure",
  "oracle_margin_minutes",
  "oracle_runtime_ms",
  "safe_structure_count",
];
const displayHeaders = [
  "Scenario ID",
  "Wind direction",
  "Wind level",
  "Charging pads K",
  "Remaining distance (m)",
  "History steps",
  "History distance (m)",
  "SOC drone 1 (%)",
  "SOC drone 2 (%)",
  "SOC drone 3 (%)",
  "SOC drone 4 (%)",
  "SOC drone 5 (%)",
  "SOC range (pp)",
  "Oracle formation + spacing",
  "Oracle drone-to-slot position",
  "Oracle total time (min)",
  "Oracle charging time (min)",
  "Second-best formation + spacing",
  "Oracle margin (min)",
  "Oracle runtime (ms)",
  "Safe structure count",
];

for (const column of selectedColumns) {
  if (!headerIndex.has(column)) throw new Error(`Missing source column: ${column}`);
}

const viewRows = rows.map((row) => selectedColumns.map((column) => row[headerIndex.get(column)]));
const view = workbook.worksheets.add("Training View");
view.getRangeByIndexes(0, 0, viewRows.length + 1, displayHeaders.length).values = [
  displayHeaders,
  ...viewRows,
];
view.showGridLines = false;
view.freezePanes.freezeRows(1);
view.freezePanes.freezeColumns(4);
const viewUsed = view.getUsedRange(true);
viewUsed.format.font = { name: "Aptos", size: 10, color: "#243447" };
view.getRange("A1:U1").format = {
  fill: "#174A67",
  font: { name: "Aptos Display", size: 10, bold: true, color: "#FFFFFF" },
  rowHeight: 34,
  wrapText: true,
  verticalAlignment: "center",
};
view.getRange(`A2:U${viewRows.length + 1}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#E7EDF2" },
};
view.getRange(`E2:E${viewRows.length + 1}`).format.numberFormat = "0.000";
view.getRange(`G2:M${viewRows.length + 1}`).format.numberFormat = "0.000";
view.getRange(`P2:S${viewRows.length + 1}`).format.numberFormat = "0.000";
view.getRange(`T2:T${viewRows.length + 1}`).format.numberFormat = "0.000";
view.getRange("A:U").format.columnWidth = 14;
view.getRange("A:A").format.columnWidth = 11;
view.getRange("B:D").format.columnWidth = 13;
view.getRange("E:E").format.columnWidth = 16;
view.getRange("N:N").format.columnWidth = 21;
view.getRange("O:O").format.columnWidth = 46;
view.getRange("P:U").format.columnWidth = 18;
const viewTable = view.tables.add(`A1:U${viewRows.length + 1}`, true, "TrainingViewTable");
viewTable.style = "TableStyleMedium2";

raw.showGridLines = false;
raw.freezePanes.freezeRows(1);
raw.freezePanes.freezeColumns(4);
raw.getUsedRange(true).format.font = { name: "Aptos", size: 9, color: "#243447" };
raw.getRange("1:1").format = {
  fill: "#375A7F",
  font: { name: "Aptos", size: 9, bold: true, color: "#FFFFFF" },
  rowHeight: 32,
  wrapText: true,
  verticalAlignment: "center",
};
raw.getUsedRange(true).format.autofitColumns();
raw.getRange("A:AV").format.columnWidth = 14;
raw.getRange("U:U").format.columnWidth = 42;

const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A1").values = [["Oracle-labelled ML Training Set"]];
summary.getRange("A1:H1").format = {
  fill: "#123B52",
  font: { name: "Aptos Display", size: 18, bold: true, color: "#FFFFFF" },
  rowHeight: 38,
  verticalAlignment: "center",
};
summary.getRange("A3:B10").values = [
  ["Dataset property", "Value"],
  ["Rows", rows.length],
  ["Wind conditions", 6],
  ["Charging-pad values", "K = 1–5"],
  ["Remaining-distance range", "0.25–25 m"],
  ["SOC range", "about 35%–100%"],
  ["Model target", "Oracle formation + spacing"],
  ["Position label retained", "Yes — oracle_position_json"],
];
summary.getRange("A3:B3").format = {
  fill: "#DCEAF2",
  font: { bold: true, color: "#123B52" },
};
summary.getRange("A3:B10").format.borders = { preset: "outside", style: "thin", color: "#9DB8C7" };

summary.getRange("D3:F3").values = [["Wind direction", "Wind level", "State count"]];
const conditions = [["head", 1], ["head", 2], ["side", 1], ["side", 2], ["tail", 1], ["tail", 2]];
summary.getRange("D4:E9").values = conditions;
summary.getRange("F4").formulas = [["=COUNTIFS('Training View'!$B$2:$B$5001,D4,'Training View'!$C$2:$C$5001,E4)"]];
summary.getRange("F4:F9").fillDown();
summary.getRange("D3:F3").format = {
  fill: "#DCEAF2",
  font: { bold: true, color: "#123B52" },
};
summary.getRange("D3:F9").format.borders = { preset: "outside", style: "thin", color: "#9DB8C7" };

const targetLabels = [...new Set(rows.map((row) => row[headerIndex.get("oracle_structure")]))].sort();
summary.getRange("A13:B13").values = [["Oracle formation + spacing", "State count"]];
summary.getRange(`A14:A${13 + targetLabels.length}`).values = targetLabels.map((label) => [label]);
summary.getRange("B14").formulas = [["=COUNTIF('Training View'!$N$2:$N$5001,A14)"]];
summary.getRange(`B14:B${13 + targetLabels.length}`).fillDown();
summary.getRange("A13:B13").format = {
  fill: "#DCEAF2",
  font: { bold: true, color: "#123B52" },
};
summary.getRange(`A13:B${13 + targetLabels.length}`).format.borders = { preset: "outside", style: "thin", color: "#9DB8C7" };
summary.getRange("A:B").format.columnWidth = 26;
summary.getRange("D:F").format.columnWidth = 16;
summary.getRange("A1:H30").format.font = { name: "Aptos", color: "#243447" };
summary.getRange("A1:H1").format.font = { name: "Aptos Display", size: 18, bold: true, color: "#FFFFFF" };

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1").values = [["How this training set was produced"]];
readme.getRange("A1:F1").format = {
  fill: "#123B52",
  font: { name: "Aptos Display", size: 18, bold: true, color: "#FFFFFF" },
  rowHeight: 38,
};
readme.getRange("A3:B8").values = [
  ["Item", "Explanation"],
  ["Experimental source", "Bideal-normalized, forward-only slot discharge rates after hover removal."],
  ["Generated state", "Wind condition, K, remaining distance, and five current SOC values produced from simulated prior flight history."],
  ["Oracle label", "The exact offline optimizer compares safe formation–spacing structures, all drone-to-slot assignments, and the K-pad charging schedule."],
  ["What is real", "The configuration-specific discharge-rate evidence comes from the processed experimental database."],
  ["What is simulated", "The 5,000 online decision states and their prior histories are simulated; they are not 5,000 physical flights."],
];
readme.getRange("A10:B15").values = [
  ["Key field", "Meaning"],
  ["soc_d1 … soc_d5", "Current reported battery levels before the next wind segment."],
  ["oracle_structure", "Best formation + inter-drone spacing under the stated objective and constraints."],
  ["oracle_position_json", "Optimal assignment of the five fixed drones to formation slots."],
  ["oracle_total_minutes", "Predicted flight time plus optimal charging makespan until all batteries reach 100%."],
  ["time__<structure>", "Oracle objective value if that formation + spacing structure is selected; blank means infeasible."],
];
for (const range of ["A3:B3", "A10:B10"]) {
  readme.getRange(range).format = { fill: "#DCEAF2", font: { bold: true, color: "#123B52" } };
}
readme.getRange("A3:B15").format = {
  font: { name: "Aptos", size: 11, color: "#243447" },
  wrapText: true,
  verticalAlignment: "top",
};
readme.getRange("A:A").format.columnWidth = 27;
readme.getRange("B:B").format.columnWidth = 92;
readme.getRange("4:8").format.rowHeight = 45;
readme.getRange("11:15").format.rowHeight = 38;

await fs.mkdir(previewDir, { recursive: true });
for (const [sheetName, range, fileName] of [
  ["Summary", "A1:F24", "summary.png"],
  ["README", "A1:B15", "readme.png"],
  ["Training View", "A1:U15", "training_view.png"],
  ["Raw Training Data", "A1:V12", "raw_training_data.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.2, format: "png" });
  await fs.writeFile(`${previewDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const check = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:F24",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 8,
});
console.log(check.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputXlsx);
console.log(`Saved ${outputXlsx}`);
