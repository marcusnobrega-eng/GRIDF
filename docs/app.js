const DATA_PATH = "data/";
const BAND_NAMES = ["K", "a", "b", "c", "R2", "RMSE", "MSE", "KS_D", "KS_p", "KS_reject", "AMD_mean", "AMD_std", "Nyears"];
const DURATION_VALUES = [5, 10, 15, 20, 25, 30, 60, 120, 240, 360, 480, 600, 720, 1440];
const RETURN_PERIODS = [2, 5, 10, 25, 50, 75, 100];
const DEFAULT_PLOT_RETURN_PERIODS = [2, 5, 10, 25, 50, 100];
const IDF_STYLE_PALETTE = ["#264653", "#287271", "#2A9D8F", "#8AB17D", "#E9C46A", "#F4A261", "#E76F51"];
const IDF_MARKERS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "triangle-down", "hexagon"];
const IDF_SERIES_STYLES = {
  2: {color: "#264653", marker: "circle"},
  5: {color: "#287271", marker: "square"},
  10: {color: "#2A9D8F", marker: "diamond"},
  25: {color: "#8AB17D", marker: "triangle-up"},
  50: {color: "#E9C46A", marker: "cross"},
  75: {color: "#F4A261", marker: "x"},
  100: {color: "#E76F51", marker: "star"}
};
const STATION_DURATIONS = ["c5", "c10", "c15", "c30", "c60", "c1440"];
const PRODUCT_OPTIONS = [
  {value: "XAVIER", en: "BR-DWGD", pt: "BR-DWGD"},
  {value: "IMERG", en: "IMERG", pt: "IMERG"},
  {value: "CHIRPS", en: "CHIRPS", pt: "CHIRPS"},
  {value: "PERSIANN", en: "PERSIANN-CDR", pt: "PERSIANN-CDR"}
];
const METHOD_OPTIONS = [
  {value: "local-interpolated", en: "Local / interpolated", pt: "Local / interpolado"},
  {value: "cetesb", en: "CETESB fixed ratios", pt: "Razões fixas CETESB"},
  {value: "station-derived", en: "Station-derived", pt: "Derivado de estações"}
];
const CITY_METHOD_OPTIONS = [
  {value: "local-interpolated", en: "Local / interpolated", pt: "Local / interpolado"}
];
const LAYER_OPTIONS = [
  {value: "intensity", en: "IDF intensity", pt: "Intensidade IDF"},
  {value: "K", en: "K (Sherman scaling)", pt: "K (escala Sherman)"},
  {value: "a", en: "a (return-period exponent)", pt: "a (expoente do período de retorno)"},
  {value: "b", en: "b (duration offset)", pt: "b (deslocamento da duração)"},
  {value: "c", en: "c (duration decay)", pt: "c (decaimento da duração)"},
  {value: "R2", en: "R² (fit quality)", pt: "R² (qualidade do ajuste)"},
  {value: "RMSE", en: "RMSE", pt: "RMSE"},
  {value: "KS_p", en: "KS p-value", pt: "p-valor KS"}
];
const DISAGG_FAMILY_OPTIONS = [
  {value: "relative_to_daily", en: "Relative to daily maximum", pt: "Relativo ao máximo diário"},
  {value: "relative_to_subdaily", en: "Relative to reference duration", pt: "Relativo à duração de referência"}
];

const COPY = {
  en: {
    brandSubtitle: "Bias-corrected gridded IDF curves for Brazil", navAtlas: "0.1° IDFs", navDisagg: "Disaggregation", navStations: "High-resolution stations", navMethods: "Data & methods", atlasKicker: "START WITH A LOCATION", atlasTitle: "Bias-corrected gridded IDF curves", atlasIntro: "Click anywhere on the map to retrieve the gridded IDF curve for the selected rainfall product and disaggregation method.", atlasControls: "IDF controls", biasCorrected: "BIAS-CORRECTED", dataset: "Rainfall dataset", disaggregationMethod: "Disaggregation method", duration: "Duration", returnPeriod: "Return period", mapLayer: "Map layer", fitBrazil: "Fit Brazil", currentSelection: "Current selection", gridResolution: "Grid", coverage: "Coverage", validPixels: "Valid pixels", quickExport: "Quick export", downloadCurrentGrid: "Download current parameter grid", gridExportNote: "Exports the selected bias-corrected Sherman parameter stack.", productNoteKicker: "PRODUCT NOTE", productNote: "This interface uses bias-corrected parameter stacks. Raw parameter stacks are intentionally excluded from this decision-support interface.", loadingRaster: "Loading gridded parameters…", mapStatus: "MAP STATUS", mapHelp: "Click the map to get a local IDF curve.", selectedLocation: "SELECTED LOCATION", noLocation: "No location selected", clickMap: "Click the map to calculate the IDF curve and design intensities at that location.", equation: "SHERMAN EQUATION", equationNote: "Intensity in mm/h; duration t in minutes.", location: "Location", source: "Source", parameters: "Parameters", designIntensities: "Design intensities", selectedDuration: "Selected duration", selectedReturnPeriod: "Selected return period", intensity: "Intensity", depth: "Depth", mmHour: "mm/h", mm: "mm", downloadWord: "Word summary", downloadCsv: "CSV values", noData: "No data", mapNoData: "No gridded value is available at this location for the selected stack.", rasterReady: "Gridded stack ready", rasterLoading: "Loading selected stack…", rasterError: "The selected gridded stack could not be loaded.", biasLabel: "Bias-corrected", resolution: "resolution", pixels: "pixels", disaggKicker: "SUPPORTING DATA", disaggTitle: "Disaggregation coefficients", disaggIntro: "Inspect the high-resolution station records used to derive the temporal-distribution ratios applied by the IDF curves.", stationStatus: "Station status", coefficientDuration: "Coefficient duration", downloadDisagg: "Download coefficient table", disaggMapNote: "Each point is a processing record from the station coefficient export.", clickStation: "Click a station to inspect its coefficient profile.", coefficientRecords: "COEFFICIENT RECORDS", coefficientTableTitle: "Station coefficient inventory", stationsKicker: "OBSERVATION SUPPORT", stationsTitle: "High-resolution stations", stationsIntro: "Review the station records that support the temporal coefficients, including accepted fits, constrained fits, and incomplete vectors.", stationInventory: "STATION INVENTORY", stationInventoryTitle: "Processing records", searchStation: "Search station ID", stationTotal: "processing rows", stationComplete: "complete vectors", stationAccepted: "accepted fits", stationWithData: "records with support", stationShown: "Showing", stationOf: "of", methodsKicker: "DOCUMENTATION", methodsTitle: "Data & methods", methodsIntro: "The browser tool exposes the bias-corrected gridded IDF curves first, then documents the supporting station and coefficient pathways.", workflowTitle: "Workflow", workflowText: "Daily rainfall products are bias-corrected using the upper-tail multiplicative factors documented in the GRIDF-BR workflow. Annual maxima are fitted with the project frequency-analysis procedure, then converted to sub-daily durations and fitted with the four-parameter Sherman equation. The interface reads the resulting K, a, b, and c rasters and calculates intensity on demand.", productTitle: "Available gridded products", productText: "The local browser bundle includes bias-corrected parameter stacks for BR-DWGD, IMERG, CHIRPS, and PERSIANN-CDR. The interface retains the three archived disaggregation pathways: local/interpolated, CETESB fixed ratios, and station-derived ratios.", stationTitle: "Station pathway", stationText: "The Disaggregation tab displays the station coefficient export. The High-resolution stations tab provides the processing inventory, fit status, record support, and CSV export.", limitsTitle: "Interpretation", limitsText: "This interface is a spatial decision-support product based on gridded model outputs. It does not replace site-specific engineering verification, and the displayed fit diagnostics should not be read as complete uncertainty bands or independent validation of the full IDF chain.", figuresKicker: "PROJECT MATERIAL", figuresTitle: "Supporting figures", figureStudy: "Study area and station inventories", figureLoocv: "Interpolation diagnostics", figureBias: "Regional coefficient differences", figureTail: "Representative tail diagnostics", welcomeKicker: "START HERE", welcomeTitle: "Get an IDF at a location", welcomeIntro: "Choose a bias-corrected rainfall product, click the IDF map, and download the resulting curve and design values.", signalAtlas: "0.1° IDFs", signalCurves: "Curves", signalExports: "Exports", welcomeStep1Title: "Choose the product", welcomeStep1Body: "Select a rainfall dataset and disaggregation method. The map is always restricted to bias-corrected parameter stacks.", welcomeStep2Title: "Click the map", welcomeStep2Body: "A click retrieves K, a, b, and c from the gridded stack and computes intensities for the selected return period and duration.", welcomeStep3Title: "Inspect and export", welcomeStep3Body: "Read the IDF curve, change the return period or duration, and download a CSV or Word summary for the selected location.", back: "Back", next: "Next", atlas: "IDF curves", statusAll: "All records", statusOk: "Accepted fit", statusViolated: "Fit with replacement or clamping", statusFailed: "No complete coefficient vector", observed: "Observed / modeled", coefficientProfile: "Coefficient profile", years: "years", observations: "observations", fitPoints: "fit points", station: "Station", status: "Status", latitude: "Latitude", longitude: "Longitude", nrmse: "NRMSE", fit: "Fit", sourceMethod: "Disaggregation", available: "available", tableEmpty: "No records match the current filter."
  },
  pt: {
    brandSubtitle: "Curvas IDF gradeadas corrigidas por viés para o Brasil", navAtlas: "0.1° IDFs", navDisagg: "Desagregação", navStations: "Estações de alta resolução", navMethods: "Dados e métodos", atlasKicker: "COMECE POR UMA LOCALIZAÇÃO", atlasTitle: "Curvas IDF gradeadas corrigidas por viés", atlasIntro: "Clique no mapa para obter a curva IDF gradeada do produto de chuva e do método de desagregação selecionados.", atlasControls: "Controles IDF", biasCorrected: "CORRIGIDO POR VIÉS", dataset: "Produto de chuva", disaggregationMethod: "Método de desagregação", duration: "Duração", returnPeriod: "Período de retorno", mapLayer: "Camada do mapa", fitBrazil: "Enquadrar Brasil", currentSelection: "Seleção atual", gridResolution: "Grade", coverage: "Cobertura", validPixels: "Pixels válidos", quickExport: "Exportação rápida", downloadCurrentGrid: "Baixar grade de parâmetros atual", gridExportNote: "Exporta a pilha de parâmetros Sherman corrigida por viés selecionada.", productNoteKicker: "NOTA DO PRODUTO", productNote: "Esta interface usa pilhas de parâmetros corrigidas por viés. As pilhas sem correção foram intencionalmente excluídas desta interface de apoio à decisão.", loadingRaster: "Carregando parâmetros gradeados…", mapStatus: "STATUS DO MAPA", mapHelp: "Clique no mapa para obter uma curva IDF local.", selectedLocation: "LOCAL SELECIONADO", noLocation: "Nenhuma localização selecionada", clickMap: "Clique no mapa para calcular a curva IDF e as intensidades de projeto no local.", equation: "EQUAÇÃO DE SHERMAN", equationNote: "Intensidade em mm/h; duração t em minutos.", location: "Local", source: "Fonte", parameters: "Parâmetros", designIntensities: "Intensidades de projeto", selectedDuration: "Duração selecionada", selectedReturnPeriod: "Período de retorno selecionado", intensity: "Intensidade", depth: "Lâmina", mmHour: "mm/h", mm: "mm", downloadWord: "Resumo Word", downloadCsv: "Valores CSV", noData: "Sem dados", mapNoData: "Não há valor gradeado disponível neste local para a pilha selecionada.", rasterReady: "Pilha gradeada pronta", rasterLoading: "Carregando pilha selecionada…", rasterError: "Não foi possível carregar a pilha gradeada selecionada.", biasLabel: "Corrigido por viés", resolution: "resolução", pixels: "pixels", disaggKicker: "DADOS DE APOIO", disaggTitle: "Coeficientes de desagregação", disaggIntro: "Inspecione os registros de estações de alta resolução usados para derivar as razões de distribuição temporal aplicadas pelas curvas IDF.", stationStatus: "Status da estação", coefficientDuration: "Duração do coeficiente", downloadDisagg: "Baixar tabela de coeficientes", disaggMapNote: "Cada ponto é um registro de processamento do exportador de coeficientes.", clickStation: "Clique em uma estação para inspecionar seu perfil de coeficientes.", coefficientRecords: "REGISTROS DE COEFICIENTES", coefficientTableTitle: "Inventário de coeficientes das estações", stationsKicker: "SUPORTE OBSERVACIONAL", stationsTitle: "Estações de alta resolução", stationsIntro: "Revise os registros de estação que apoiam os coeficientes temporais, incluindo ajustes aceitos, ajustes com restrição e vetores incompletos.", stationInventory: "INVENTÁRIO DE ESTAÇÕES", stationInventoryTitle: "Registros de processamento", searchStation: "Buscar ID da estação", stationTotal: "linhas processadas", stationComplete: "vetores completos", stationAccepted: "ajustes aceitos", stationWithData: "registros com suporte", stationShown: "Mostrando", stationOf: "de", methodsKicker: "DOCUMENTAÇÃO", methodsTitle: "Dados e métodos", methodsIntro: "A ferramenta expõe primeiro as curvas IDF gradeadas corrigidas por viés e depois documenta os caminhos de estações e coeficientes.", workflowTitle: "Fluxo", workflowText: "Os produtos diários de chuva são corrigidos por viés usando os fatores multiplicativos da cauda superior documentados no fluxo GRIDF-BR. Os máximos anuais são ajustados pelo procedimento de análise de frequência do projeto, convertidos para durações subdiárias e ajustados pela equação Sherman de quatro parâmetros. A interface lê as grades K, a, b e c e calcula a intensidade sob demanda.", productTitle: "Produtos gradeados disponíveis", productText: "O pacote local inclui pilhas de parâmetros corrigidas por viés para BR-DWGD, IMERG, CHIRPS e PERSIANN-CDR. A interface mantém os três caminhos de desagregação arquivados: local/interpolado, razões fixas CETESB e derivado de estações.", stationTitle: "Caminho das estações", stationText: "A aba Desagregação exibe o exportador de coeficientes das estações. A aba Estações de alta resolução fornece o inventário de processamento, o status do ajuste, o suporte do registro e a exportação CSV.", limitsTitle: "Interpretação", limitsText: "Esta interface é um produto espacial de apoio à decisão baseado em resultados de modelos gradeados. Ele não substitui a verificação de engenharia local, e os diagnósticos de ajuste exibidos não devem ser lidos como bandas completas de incerteza ou validação independente de toda a cadeia IDF.", figuresKicker: "MATERIAL DO PROJETO", figuresTitle: "Figuras de apoio", figureStudy: "Área de estudo e inventários de estações", figureLoocv: "Diagnósticos da interpolação", figureBias: "Diferenças regionais dos coeficientes", figureTail: "Diagnósticos representativos da cauda", welcomeKicker: "COMECE AQUI", welcomeTitle: "Obtenha uma IDF em um local", welcomeIntro: "Escolha um produto de chuva corrigido por viés, clique no mapa IDF e baixe a curva e os valores de projeto.", signalAtlas: "0.1° IDFs", signalCurves: "Curvas", signalExports: "Exportações", welcomeStep1Title: "Escolha o produto", welcomeStep1Body: "Selecione um produto de chuva e um método de desagregação. O mapa usa apenas pilhas de parâmetros corrigidas por viés.", welcomeStep2Title: "Clique no mapa", welcomeStep2Body: "O clique recupera K, a, b e c da pilha gradeada e calcula intensidades para o período de retorno e a duração selecionados.", welcomeStep3Title: "Inspecione e exporte", welcomeStep3Body: "Leia a curva IDF, altere o período de retorno ou a duração e baixe um resumo CSV ou Word do local selecionado.", back: "Voltar", next: "Avançar", atlas: "Curvas IDF", statusAll: "Todos os registros", statusOk: "Ajuste aceito", statusViolated: "Ajuste com substituição ou restrição", statusFailed: "Sem vetor completo", observed: "Observado / modelado", coefficientProfile: "Perfil dos coeficientes", years: "anos", observations: "observações", fitPoints: "pontos de ajuste", station: "Estação", status: "Status", latitude: "Latitude", longitude: "Longitude", nrmse: "NRMSE", fit: "Ajuste", sourceMethod: "Desagregação", available: "disponível", tableEmpty: "Nenhum registro corresponde ao filtro atual."
  }
};

const FIGURES = [
  {file: "assets/study-area.png", en: "Study area and station inventories", pt: "Área de estudo e inventários de estações", descEn: "Station support and product context used by the project.", descPt: "Suporte das estações e contexto dos produtos usados no projeto."},
  {file: "assets/loocv-summary.png", en: "Interpolation diagnostics", pt: "Diagnósticos da interpolação", descEn: "Duration-specific diagnostics for the interpolated coefficients.", descPt: "Diagnósticos por duração dos coeficientes interpolados."},
  {file: "assets/biome-bias.png", en: "Regional coefficient differences", pt: "Diferenças regionais dos coeficientes", descEn: "GRIDF ratios relative to the CETESB reference.", descPt: "Razões GRIDF em relação à referência CETESB."},
  {file: "assets/tail-diagnostics.png", en: "Representative tail diagnostics", pt: "Diagnósticos representativos da cauda", descEn: "Selected annual-maximum tail diagnostics.", descPt: "Diagnósticos selecionados da cauda dos máximos anuais."}
];

FIGURES.push({file: "assets/gauge-network.png", en: "Gauge network and coefficient support", pt: "Rede de estações e suporte dos coeficientes", descEn: "Observed station coverage supporting the temporal-distribution coefficients.", descPt: "Cobertura das estações observadas que apoia os coeficientes de distribuição temporal."});

Object.assign(COPY.en, {coefficientFamily: "Coefficient family", interpolationMethod: "INTERPOLATION", disaggControlNote: "The selected surface is an interpolated coefficient grid; station records are shown below for context.", disaggMapHelp: "Click a station to inspect its observed coefficient vector.", disaggTableNote: "A compact sample of station records is shown here; the complete station inventory is available in the High-resolution stations tab."});
Object.assign(COPY.pt, {coefficientFamily: "Família do coeficiente", interpolationMethod: "INTERPOLAÇÃO", disaggControlNote: "A superfície selecionada é uma grade de coeficientes interpolada; os registros das estações são mostrados abaixo como contexto.", disaggMapHelp: "Clique em uma estação para inspecionar seu vetor observado de coeficientes.", disaggTableNote: "Uma amostra compacta dos registros de estação é mostrada aqui; o inventário completo está na aba Estações de alta resolução."});
COPY.en.productText = "The local browser bundle includes bias-corrected parameter stacks for BR-DWGD, IMERG, CHIRPS, and PERSIANN-CDR. The interface retains the three archived disaggregation pathways: local/interpolated, CETESB fixed ratios, and station-derived ratios.";
COPY.pt.productText = "O pacote local inclui pilhas de parâmetros corrigidas por viés para BR-DWGD, IMERG, CHIRPS e PERSIANN-CDR. A interface mantém os três caminhos de desagregação arquivados: local/interpolado, razões fixas CETESB e derivado de estações.";
Object.assign(COPY.en, {
  cityMethodNote: "Corrected annual maxima are area-weighted over each municipality, then Gumbel return levels are refitted by the method of moments and the Sherman relation is refitted at municipal scale.",
  cityCatalogDownloadNote: "Exports all municipalities for the selected product, duration, and return period. Municipal frequency analysis uses Gumbel fitted by the method of moments.",
  citySourceNote: "Municipal annual maxima were aggregated from corrected raster series using polygon-cell intersection areas; Gumbel return levels were fitted by the method of moments and Sherman parameters were then refitted.",
  cityReportWorkflow: "The selected product is fitted with a Gumbel distribution by the method of moments using the available municipal annual values. Return depths are converted to sub-daily durations with area-weighted local/interpolated temporal ratios, and the four-parameter Sherman relation is refitted to those municipal intensities.",
  cityReportFrequency: "Frequency analysis",
  cityReportFrequencyText: "Municipal IDFs use the Gumbel distribution fitted by the method of moments for all city-scale frequency estimates.",
  cityReportDisaggDailyFigure: "Coefficients relative to the daily maximum",
  cityReportDisaggReferenceFigure: "Coefficients relative to the reference duration"
});
Object.assign(COPY.pt, {
  cityMethodNote: "Os máximos anuais corrigidos são ponderados pela área de cada município; os níveis de retorno Gumbel são reajustados pelo método dos momentos e a relação Sherman é reajustada na escala municipal.",
  cityCatalogDownloadNote: "Exporta todos os municípios para o produto, duração e período de retorno selecionados. A análise de frequência municipal usa Gumbel ajustado pelo método dos momentos.",
  citySourceNote: "Os máximos anuais municipais foram agregados das grades corrigidas usando as áreas de interseção entre polígonos e células; os níveis de retorno Gumbel foram ajustados pelo método dos momentos e os parâmetros Sherman foram então reajustados.",
  cityReportWorkflow: "O produto selecionado é ajustado pela distribuição Gumbel, pelo método dos momentos, usando os valores anuais municipais disponíveis. As lâminas de retorno são convertidas para durações subdiárias com razões temporais locais/interpoladas ponderadas pela área, e a relação Sherman de quatro parâmetros é reajustada para essas intensidades municipais.",
  cityReportFrequency: "Análise de frequência",
  cityReportFrequencyText: "As IDFs municipais utilizam a distribuição Gumbel ajustada pelo método dos momentos para todas as estimativas de frequência em escala municipal.",
  cityReportDisaggDailyFigure: "Coeficientes relativos ao máximo diário",
  cityReportDisaggReferenceFigure: "Coeficientes relativos à duração de referência"
});

const state = {lang: "pt", view: "atlas", product: "XAVIER", method: "local-interpolated", duration: 60, returnPeriod: 10, plotReturnPeriods: DEFAULT_PLOT_RETURN_PERIODS.slice(), layer: "intensity", data: {}, rasterCache: new Map(), raster: null, overlay: null, overlayToken: 0, map: null, disaggMap: null, disaggFamily: "relative_to_daily", disaggDuration: 60, disaggRasterCache: new Map(), disaggRaster: null, disaggOverlay: null, disaggOverlayToken: 0, boundary: null, states: null, disaggBoundary: null, stationLayer: null, selected: null, disaggSelected: null, welcomeStep: 0};
const $ = (id) => document.getElementById(id);
state.lang = "pt";
const t = (key) => COPY[state.lang][key] || key;
const fmt = (value, digits = 2) => value == null || !Number.isFinite(Number(value)) ? t("noData") : Number(value).toLocaleString(state.lang === "pt" ? "pt-BR" : "en-US", {maximumFractionDigits: digits});
const isFiniteNumber = (value) => value != null && Number.isFinite(Number(value));
const formatDuration = (minutes) => Number(minutes) === 1440 ? "24 h" : Number(minutes) >= 60 ? `${Number(minutes) / 60} h` : `${minutes} min`;
const formatDurationKey = (key) => ({c5: "5 min", c10: "10 min", c15: "15 min", c30: "30 min", c60: "1 h", c1440: "24 h"}[key] || key);
const statusText = (status) => ({"ok-fit": t("statusOk"), "violated-fit": t("statusViolated"), failed: t("statusFailed")} [status] || status);

async function loadJson(name) { const response = await fetch(`${DATA_PATH}${name}`); if (!response.ok) throw new Error(`${name} (${response.status})`); return response.json(); }
function showToast(message) { const toast = $("toast"); toast.textContent = message; toast.classList.add("show"); clearTimeout(showToast.timer); showToast.timer = setTimeout(() => toast.classList.remove("show"), 2800); }
function refreshIcons() { if (window.lucide) window.lucide.createIcons(); }
function downloadBlob(filename, blob) { const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 700); }
function csvEscape(value) { const text = value == null ? "" : String(value); return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text; }
function downloadCsv(filename, rows) { if (!rows.length) { showToast(t("tableEmpty")); return; } const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))]; const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => csvEscape(row[column])).join(","))].join("\n"); downloadBlob(filename, new Blob([csv], {type: "text/csv;charset=utf-8"})); showToast(t("available")); }
function xmlEscape(value) { return String(value ?? "").replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;"); }
function docParagraph(text, style = "Normal") { return `<w:p><w:pPr><w:pStyle w:val="${style}"/></w:pPr><w:r><w:t xml:space="preserve">${xmlEscape(text)}</w:t></w:r></w:p>`; }
function docTable(rows) { return `<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="5" w:color="B7C8C3"/><w:left w:val="single" w:sz="5" w:color="B7C8C3"/><w:bottom w:val="single" w:sz="5" w:color="B7C8C3"/><w:right w:val="single" w:sz="5" w:color="B7C8C3"/><w:insideH w:val="single" w:sz="4" w:color="D6DFDC"/><w:insideV w:val="single" w:sz="4" w:color="D6DFDC"/></w:tblBorders></w:tblPr>${rows.map((row, i) => `<w:tr>${row.map((cell) => `<w:tc><w:tcPr>${i === 0 ? '<w:shd w:fill="176B78"/>' : ""}</w:tcPr><w:p><w:r>${i === 0 ? '<w:rPr><w:b/><w:color w:val="FFFFFF"/></w:rPr>' : ""}<w:t>${xmlEscape(cell)}</w:t></w:r></w:p></w:tc>`).join("")}</w:tr>`).join("")}</w:tbl>`; }
async function downloadWord(filename, title, paragraphs, rows) { try { const body = [docParagraph("GRIDF-BR | IDF curves", "Eyebrow"), docParagraph(title, "Title"), ...paragraphs.map((item) => docParagraph(item)), rows?.length ? docTable(rows) : "", '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900"/></w:sectPr>'].join(""); const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">${body}</w:document>`; const styles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:color w:val="16333F"/><w:sz w:val="21"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Eyebrow"><w:name w:val="Eyebrow"/><w:rPr><w:b/><w:color w:val="176B78"/><w:sz w:val="18"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:color w:val="176B78"/><w:sz w:val="34"/></w:rPr></w:style></w:styles>`; const zip = new JSZip(); zip.file("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'); zip.folder("_rels").file(".rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'); zip.folder("word").file("document.xml", documentXml); zip.folder("word").file("styles.xml", styles); zip.folder("word").folder("_rels").file("document.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'); downloadBlob(filename, await zip.generateAsync({type: "blob", compression: "DEFLATE"})); showToast(t("available")); } catch (error) { console.error(error); showToast(t("rasterError")); } }

function docHeading(text, style = "Heading1") { return `<w:p><w:pPr><w:pStyle w:val="${style}"/></w:pPr><w:r><w:t>${xmlEscape(text)}</w:t></w:r></w:p>`; }
function docReportSection(heading, paragraphs = [], bullets = []) { return [docHeading(heading), ...paragraphs.map((item) => docParagraph(item)), ...bullets.map((item) => docParagraph(`- ${item}`))].join(""); }
async function downloadWordReport(filename, title, metadata, sections, rows) { try { const body = [docParagraph("GRIDF-BR | Bias-corrected gridded IDF curves", "Eyebrow"), docParagraph(title, "Title"), docParagraph(metadata, "Subtitle"), ...sections.map((section) => docReportSection(section.heading, section.paragraphs, section.bullets)), rows?.length ? docHeading(t("designIntensities"), "Heading1") + docTable(rows) : "", '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="900" w:right="900" w:bottom="900" w:left="900"/></w:sectPr>'].join(""); const documentXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">${body}</w:document>`; const styles = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:color w:val="16333F"/><w:sz w:val="21"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Eyebrow"><w:name w:val="Eyebrow"/><w:rPr><w:b/><w:color w:val="176B78"/><w:sz w:val="18"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:color w:val="176B78"/><w:sz w:val="34"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:i/><w:color w:val="647B82"/><w:sz w:val="21"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Georgia" w:hAnsi="Georgia"/><w:b/><w:color w:val="176B78"/><w:sz w:val="25"/></w:rPr></w:style></w:styles>`; const zip = new JSZip(); zip.file("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'); zip.folder("_rels").file(".rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'); zip.folder("word").file("document.xml", documentXml); zip.folder("word").file("styles.xml", styles); zip.folder("word").folder("_rels").file("document.xml.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'); downloadBlob(filename, await zip.generateAsync({type: "blob", compression: "DEFLATE"})); showToast(t("available")); } catch (error) { console.error(error); showToast(t("rasterError")); } }

function selectedCatalogEntry() { return state.data.idfCatalog.records.find((entry) => entry.product === state.product && entry.method === state.method); }
function selectedProductLabel() { return PRODUCT_OPTIONS.find((item) => item.value === state.product)?.[state.lang] || state.product; }
function selectedMethodLabel() { return METHOD_OPTIONS.find((item) => item.value === state.method)?.[state.lang] || state.method; }
function fillSelect(id, items, selected, labelKey = "en") { const select = $(id); select.innerHTML = items.map((item) => `<option value="${item.value}">${item[labelKey] || item.label}</option>`).join(""); select.value = selected; }
function populateSelects() { fillSelect("idfProduct", PRODUCT_OPTIONS, state.product, state.lang); fillSelect("idfMethod", METHOD_OPTIONS, state.method, state.lang); fillSelect("idfLayer", LAYER_OPTIONS, state.layer, state.lang); fillSelect("idfDuration", DURATION_VALUES.map((value) => ({value, label: formatDuration(value)})), state.duration); fillSelect("idfReturnPeriod", RETURN_PERIODS.map((value) => ({value, label: `${value} yr`})), state.returnPeriod); fillSelect("disaggStatus", [{value: "all", label: t("statusAll")}, {value: "ok-fit", label: t("statusOk")}, {value: "violated-fit", label: t("statusViolated")}, {value: "failed", label: t("statusFailed")}], state.disaggStatus || "all"); fillSelect("disaggDuration", STATION_DURATIONS.map((value) => ({value, label: formatDurationKey(value)})), state.disaggDuration || "c5"); fillSelect("stationStatus", [{value: "all", label: t("statusAll")}, {value: "ok-fit", label: t("statusOk")}, {value: "violated-fit", label: t("statusViolated")}, {value: "failed", label: t("statusFailed")}], state.stationStatus || "all"); }
function setLanguage(language) { state.lang = language; document.documentElement.lang = language; document.querySelectorAll("[data-i18n]").forEach((element) => { const key = element.dataset.i18n; if (COPY[language][key]) element.textContent = COPY[language][key]; }); document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => { element.placeholder = COPY[language][element.dataset.i18nPlaceholder] || element.placeholder; }); document.querySelectorAll(".language-button").forEach((button) => { const active = button.dataset.language === language; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); }); populateSelects(); renderAtlasMeta(); if (state.selected) renderIdfDetail(); if (state.view === "disagg") { renderDisaggMap(); renderDisaggDetail(); renderDisaggTable(); } if (state.view === "stations") renderStationSummary(); if (state.view === "methods") renderMethods(); refreshIcons(); }

async function loadRaster(entry) { const key = `${entry.product}|${entry.method}`; if (state.rasterCache.has(key)) return state.rasterCache.get(key); if (!window.GeoTIFF) throw new Error("GeoTIFF library unavailable"); const response = await fetch(`${DATA_PATH}${entry.file.replace(/^.*?data\//, "")}`); if (!response.ok) throw new Error(`${entry.file} (${response.status})`); const tiff = await GeoTIFF.fromArrayBuffer(await response.arrayBuffer()); const image = await tiff.getImage(); const arrays = await image.readRasters({interleave: false}); const raster = {entry, image, arrays: Array.isArray(arrays) ? arrays : [arrays], width: image.getWidth(), height: image.getHeight(), bounds: image.getBoundingBox(), resolution: image.getResolution()}; state.rasterCache.set(key, raster); return raster; }
function rasterPixel(raster, band, x, y) { if (x < 0 || y < 0 || x >= raster.width || y >= raster.height) return null; const value = raster.arrays[band][y * raster.width + x]; return Number.isFinite(Number(value)) ? Number(value) : null; }
function rasterParametersAt(raster, x, y) { const values = Object.fromEntries(["K", "a", "b", "c", "R2", "RMSE", "MSE", "KS_D", "KS_p", "KS_reject", "AMD_mean", "AMD_std", "Nyears"].map((name) => [name, rasterPixel(raster, BAND_NAMES.indexOf(name), x, y)])); return [values.K, values.a, values.b, values.c].every(isFiniteNumber) ? values : null; }
function rasterCellValue(raster, layer, x, y) { if (layer === "intensity") { const params = rasterParametersAt(raster, x, y); if (!params) return null; const value = idfIntensity(params, state.duration, state.returnPeriod); return Number.isFinite(Number(value)) && Number(value) >= 0 ? Number(value) : null; } const value = rasterPixel(raster, BAND_NAMES.indexOf(layer), x, y); return isFiniteNumber(value) ? Number(value) : null; }
function rasterValuesForLayer(raster, layer) { const values = new Float32Array(raster.width * raster.height); values.fill(NaN); for (let y = 0; y < raster.height; y += 1) for (let x = 0; x < raster.width; x += 1) { const value = rasterCellValue(raster, layer, x, y); if (value != null) values[y * raster.width + x] = value; } return values; }
function quantile(values, q) { const sample = []; const step = Math.max(1, Math.floor(values.length / 50000)); for (let i = 0; i < values.length; i += step) if (Number.isFinite(values[i])) sample.push(values[i]); if (!sample.length) return 0; sample.sort((a, b) => a - b); return sample[Math.min(sample.length - 1, Math.floor((sample.length - 1) * q))]; }
function colorFor(value, min, max) { const stops = [[20, 59, 90], [23, 107, 120], [85, 185, 159], [240, 190, 98], [207, 90, 83]]; const ratio = Math.max(0, Math.min(1, (value - min) / (max - min || 1))); const scaled = ratio * (stops.length - 1); const left = Math.floor(scaled); const right = Math.min(stops.length - 1, left + 1); const part = scaled - left; return stops[left].map((item, index) => Math.round(item + (stops[right][index] - item) * part)); }
function layerScale(entry, layer, values) { if (layer === "intensity") return {min: quantile(values, .02), max: quantile(values, .98)}; if (layer === "KS_p") return {min: 0, max: 1}; if (layer === "KS_reject") return {min: 0, max: 1}; const stats = entry.bandStats[layer] || {}; return {min: isFiniteNumber(stats.p02) ? Number(stats.p02) : quantile(values, .02), max: isFiniteNumber(stats.p98) ? Number(stats.p98) : quantile(values, .98)}; }
function layerLabel() { return LAYER_OPTIONS.find((item) => item.value === state.layer)?.[state.lang] || state.layer; }
function createProjectedRasterLayer(raster, values, scale) {
  const colors = new Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) continue;
    const rgb = colorFor(values[i], scale.min, scale.max);
    colors[i] = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
  }
  return L.Layer.extend({
    onAdd(map) {
      this._map = map;
      this._canvas = L.DomUtil.create("canvas", "leaflet-layer gridf-raster-overlay");
      this._canvas.style.position = "absolute";
      this._canvas.style.opacity = ".78";
      this._canvas.style.pointerEvents = "none";
      this._context = this._canvas.getContext("2d");
      map.getPanes().overlayPane.appendChild(this._canvas);
      map.on("moveend zoomend resize", this._scheduleReset, this);
      this._reset();
    },
    onRemove(map) {
      map.off("moveend zoomend resize", this._scheduleReset, this);
      if (this._frame) {
        L.Util.cancelAnimFrame(this._frame);
        this._frame = null;
      }
      L.DomUtil.remove(this._canvas);
    },
    _scheduleReset() {
      if (this._frame) L.Util.cancelAnimFrame(this._frame);
      this._frame = L.Util.requestAnimFrame(this._reset, this);
    },
    _reset() {
      this._frame = null;
      const map = this._map;
      if (!map || !this._canvas || !this._context) return;
      const size = map.getSize();
      const topLeft = map.containerPointToLayerPoint([0, 0]);
      L.DomUtil.setPosition(this._canvas, topLeft);
      this._canvas.width = size.x;
      this._canvas.height = size.y;
      this._draw(topLeft);
    },
    _draw(topLeft) {
      const context = this._context;
      const map = this._map;
      const [west, south, east, north] = raster.bounds.map(Number);
      const dx = (east - west) / raster.width;
      const dy = (north - south) / raster.height;
      const view = map.getBounds().pad(0.08);
      const xStart = Math.max(0, Math.floor((view.getWest() - west) / dx) - 1);
      const xEnd = Math.min(raster.width - 1, Math.ceil((view.getEast() - west) / dx) + 1);
      const yStart = Math.max(0, Math.floor((north - view.getNorth()) / dy) - 1);
      const yEnd = Math.min(raster.height - 1, Math.ceil((north - view.getSouth()) / dy) + 1);

      context.clearRect(0, 0, this._canvas.width, this._canvas.height);
      context.globalAlpha = 1;
      for (let y = yStart; y <= yEnd; y += 1) {
        const latTop = north - y * dy;
        const latBottom = north - (y + 1) * dy;
        for (let x = xStart; x <= xEnd; x += 1) {
          const color = colors[y * raster.width + x];
          if (!color) continue;
          const lonLeft = west + x * dx;
          const lonRight = west + (x + 1) * dx;
          const nw = map.latLngToLayerPoint([latTop, lonLeft]);
          const se = map.latLngToLayerPoint([latBottom, lonRight]);
          context.fillStyle = color;
          context.fillRect(
            Math.floor(nw.x - topLeft.x),
            Math.floor(nw.y - topLeft.y),
            Math.ceil(se.x - nw.x) + 1,
            Math.ceil(se.y - nw.y) + 1
          );
        }
      }
    }
  });
}
async function renderRaster() {
  const token = ++state.overlayToken;
  const entry = selectedCatalogEntry();
  if (!entry) return;
  $("rasterStatus").textContent = t("rasterLoading");
  $("mapStatusText").textContent = t("rasterLoading");
  try {
    const raster = await loadRaster(entry);
    if (token !== state.overlayToken) return;
    state.raster = raster;
    const values = rasterValuesForLayer(raster, state.layer);
    const scale = layerScale(entry, state.layer, values);

    if (state.overlay) state.overlay.remove();
    const RasterLayer = createProjectedRasterLayer(raster, values, scale);
    state.overlay = new RasterLayer().addTo(state.map);
    $("mapLayerTitle").textContent = layerLabel();
    $("mapStatusText").textContent = `${selectedProductLabel()} · ${selectedMethodLabel()}`;
    $("rasterStatus").textContent = `${t("rasterReady")} · ${selectedProductLabel()}`;
    $("gridResolution").textContent = `${Math.abs(raster.resolution[0]).toFixed(2)}°`;
    $("gridCoverage").textContent = `${raster.width} × ${raster.height}`;
    $("gridPixels").textContent = `${fmt(entry.bandStats.K.valid, 0)} ${t("available")}`;
    $("rasterLegend").innerHTML = `<div class="legend-title">${layerLabel()}${state.layer === "intensity" ? ` · ${state.returnPeriod} yr · ${formatDuration(state.duration)}` : ""}</div><div class="legend-bar"></div><div class="legend-labels"><span>${fmt(scale.min, state.layer === "intensity" ? 1 : 2)}</span><span>${fmt(scale.max, state.layer === "intensity" ? 1 : 2)}</span></div>`;
  } catch (error) {
    console.error(error);
    $("rasterStatus").textContent = t("rasterError");
    $("mapStatusText").textContent = t("rasterError");
    showToast(t("rasterError"));
  }
}
function sampleRaster(raster, lat, lon) { const [west, south, east, north] = raster.bounds; const x = Math.floor((lon - west) / raster.resolution[0]); const y = Math.floor((north - lat) / Math.abs(raster.resolution[1])); if (lon < west || lon > east || lat < south || lat > north) return null; const values = Object.fromEntries(BAND_NAMES.map((name, index) => [name, rasterPixel(raster, index, x, y)])); if (![values.K, values.a, values.b, values.c].every(isFiniteNumber)) return null; return values; }
function idfIntensity(params, duration, returnPeriod) { if (!params || ![params.K, params.a, params.b, params.c].every(isFiniteNumber)) return null; const value = Number(params.K) * Math.pow(Number(returnPeriod), Number(params.a)) / Math.pow(Number(params.b) + Number(duration), Number(params.c)); return Number.isFinite(value) && value >= 0 ? value : null; }
function curveRows(params) { return DURATION_VALUES.map((duration) => { const row = {duration: formatDuration(duration), duration_min: duration}; RETURN_PERIODS.forEach((period) => { row[`rp_${period}_yr_mm_h`] = idfIntensity(params, duration, period); row[`rp_${period}_yr_mm`] = idfIntensity(params, duration, period) * duration / 60; }); return row; }); }
function renderIdfDetail() { const selected = state.selected; if (!selected) return; const params = selected.params; const rows = curveRows(params); const values = rows.find((row) => row.duration_min === Number(state.duration)); const entry = selected.entry; $("idfDetail").innerHTML = `<div class="detail-header"><div><span class="eyebrow">${t("selectedLocation")}</span><h2>${fmt(selected.lat, 4)}, ${fmt(selected.lon, 4)}</h2><p class="detail-meta">${selectedProductLabel()} · ${selectedMethodLabel()}</p></div><span class="status-tag">${t("biasLabel")}</span></div><div class="detail-source"><strong>${t("source")}</strong><br/>${entry.productLabel} · ${entry.methodLabel} · ${entry.width} × ${entry.height} · ${Math.abs(entry.transform[0]).toFixed(2)}°</div><h3 class="detail-heading">${t("parameters")}</h3><div class="parameter-grid">${["K", "a", "b", "c"].map((key) => `<div class="parameter-card"><span>${key}</span><strong>${fmt(params[key], key === "K" ? 1 : 3)}</strong></div>`).join("")}</div><div class="parameter-grid">${["R2", "RMSE", "KS_p", "Nyears"].map((key) => `<div class="parameter-card"><span>${key === "KS_p" ? "KS p" : key}</span><strong>${fmt(params[key], key === "Nyears" ? 0 : 3)}</strong></div>`).join("")}</div><h3 class="detail-heading">${t("designIntensities")}</h3><div id="idfCurve" class="chart-box"></div><p class="small-note">${t("selectedReturnPeriod")}: ${state.returnPeriod} yr · ${t("selectedDuration")}: ${formatDuration(state.duration)}</p><div class="table-wrap"><table><thead><tr><th>${t("duration")}</th><th>${t("intensity")} (${t("mmHour")})</th><th>${t("depth")} (${t("mm")})</th></tr></thead><tbody>${[5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => { const intensity = idfIntensity(params, duration, state.returnPeriod); return `<tr><td>${formatDuration(duration)}</td><td class="table-number">${fmt(intensity, 2)}</td><td class="table-number">${fmt(intensity * duration / 60, 2)}</td></tr>`; }).join("")}</tbody></table></div><div class="detail-actions"><button class="action-button primary" id="downloadSelectedWord" type="button"><i data-lucide="file-text" aria-hidden="true"></i>${t("downloadWord")}</button><button class="action-button secondary" id="downloadSelectedCsv" type="button"><i data-lucide="download" aria-hidden="true"></i>${t("downloadCsv")}</button></div>`; $("downloadSelectedWord").onclick = () => { const rowsForWord = [[t("duration"), `${t("intensity")} (${t("mmHour")})`, `${t("depth")} (${t("mm")})`], ...[5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => { const intensity = idfIntensity(params, duration, state.returnPeriod); return [formatDuration(duration), fmt(intensity, 2), fmt(intensity * duration / 60, 2)]; })]; downloadWord("gridf_idf_summary.docx", `GRIDF-BR IDF · ${fmt(selected.lat, 4)}, ${fmt(selected.lon, 4)}`, [`${selectedProductLabel()} · ${selectedMethodLabel()} · ${t("biasLabel")}`, `K=${fmt(params.K, 3)}, a=${fmt(params.a, 4)}, b=${fmt(params.b, 3)}, c=${fmt(params.c, 4)}`, `${t("returnPeriod")}: ${state.returnPeriod} yr`], rowsForWord); }; $("downloadSelectedCsv").onclick = () => downloadCsv("gridf_idf_values.csv", rows.map((row) => ({latitude: selected.lat, longitude: selected.lon, product: entry.product, disaggregation: entry.method, bias: "BC", ...row}))); refreshIcons(); if (window.Plotly) { const series = RETURN_PERIODS.map((period, index) => ({x: rows.map((row) => row.duration), y: rows.map((row) => row[`rp_${period}_yr_mm_h`]), type: "scatter", mode: "lines+markers", name: `${period} yr`, line: {width: 2.2}, marker: {size: 5}})); Plotly.newPlot("idfCurve", series, {margin: {l: 46, r: 8, t: 8, b: 50}, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {title: state.lang === "pt" ? "Duração" : "Duration", tickangle: -35}, yaxis: {title: state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", rangemode: "tozero"}, legend: {orientation: "h", y: 1.2}, font: {family: "DM Sans", size: 9, color: "#16333f"}}, {displayModeBar: false, responsive: true}); } }
async function handleAtlasClick(event) { if (!state.raster) return; const params = sampleRaster(state.raster, event.latlng.lat, event.latlng.lng); if (!params) { showToast(t("mapNoData")); return; } state.selected = {lat: event.latlng.lat, lon: event.latlng.lng, params, entry: state.raster.entry}; renderIdfDetail(); if (window.innerWidth <= 1020) $("idfDetailPanel").classList.add("open"); }
function renderAtlasMeta() {
  if (!state.raster) return;
  const entry = selectedCatalogEntry();
  $("mapLayerTitle").textContent = layerLabel();
  $("mapStatusText").textContent = `${selectedProductLabel()} · ${selectedMethodLabel()}`;
  $("rasterStatus").textContent = `${t("rasterReady")} · ${selectedProductLabel()}`;
  $("gridResolution").textContent = `${Math.abs(state.raster.resolution[0]).toFixed(2)}°`;
  $("gridCoverage").textContent = `${state.raster.width} × ${state.raster.height}`;
  if (entry?.bandStats?.K) $("gridPixels").textContent = `${fmt(entry.bandStats.K.valid, 0)} ${t("available")}`;
  $("rasterLegend").querySelector(".legend-title")?.replaceChildren(document.createTextNode(`${layerLabel()}${state.layer === "intensity" ? ` · ${state.returnPeriod} yr · ${formatDuration(state.duration)}` : ""}`));
}
function initAtlasMap() { state.map = L.map("idfMap", {zoomControl: false, preferCanvas: true}).setView([-14.2, -52.5], 4); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "&copy; OpenStreetMap contributors", maxZoom: 11}).addTo(state.map); state.map.createPane("atlasMunicipalPane"); state.map.getPane("atlasMunicipalPane").style.zIndex = 405; state.map.getPane("atlasMunicipalPane").style.pointerEvents = "none"; state.map.createPane("atlasBoundaryPane"); state.map.getPane("atlasBoundaryPane").style.zIndex = 430; state.map.getPane("atlasBoundaryPane").style.pointerEvents = "none"; L.control.zoom({position: "bottomleft"}).addTo(state.map); state.boundary = L.geoJSON(state.data.brazil, {pane: "atlasBoundaryPane", style: {color: "#16333f", weight: 1.5, fillColor: "#75a9a0", fillOpacity: .08}}).addTo(state.map); state.states = L.geoJSON(state.data.states, {pane: "atlasBoundaryPane", style: {color: "#5f8885", weight: .35, opacity: .55, fillOpacity: 0}}).addTo(state.map); state.map.on("click", handleAtlasClick); $("zoomInButton").onclick = () => state.map.zoomIn(); $("zoomOutButton").onclick = () => state.map.zoomOut(); $("fitBrazilButton").onclick = () => state.map.fitBounds(state.boundary.getBounds(), {padding: [14, 14]}); setTimeout(() => state.map.fitBounds(state.boundary.getBounds(), {padding: [14, 14]}), 250); }
function getFilteredStations(status, search = "") { const normalized = search.trim().toLowerCase(); return state.data.stations.records.filter((row) => (status === "all" || row.status === status) && (!normalized || String(row.id).toLowerCase().includes(normalized))); }
function stationMarkerColor(row) { return row.status === "ok-fit" ? "#2d8c67" : row.status === "violated-fit" ? "#e88b3a" : "#cf5a53"; }
function initDisaggMap() { state.disaggMap = L.map("disaggMap", {zoomControl: false, preferCanvas: true}).setView([-14.2, -52.5], 4); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "&copy; OpenStreetMap contributors", maxZoom: 11}).addTo(state.disaggMap); L.control.zoom({position: "bottomleft"}).addTo(state.disaggMap); state.disaggBoundary = L.geoJSON(state.data.brazil, {style: {color: "#16333f", weight: 1.3, fillColor: "#75a9a0", fillOpacity: .08}}).addTo(state.disaggMap); state.stationLayer = L.layerGroup().addTo(state.disaggMap); state.disaggMap.fitBounds(state.disaggBoundary.getBounds(), {padding: [14, 14]}); }
function renderDisaggMap() { if (!state.stationLayer) return; const rows = getFilteredStations(state.disaggStatus || "all"); state.stationLayer.clearLayers(); rows.forEach((row) => { if (!isFiniteNumber(row.lat) || !isFiniteNumber(row.lon)) return; const marker = L.circleMarker([row.lat, row.lon], {radius: 3.8, color: stationMarkerColor(row), weight: .9, fillColor: stationMarkerColor(row), fillOpacity: .72}); marker.bindTooltip(`${row.id} · ${statusText(row.status)}`, {direction: "top", offset: [0, -3]}); marker.on("click", () => { state.disaggSelected = row; renderDisaggDetail(); }); marker.addTo(state.stationLayer); }); $("disaggMapStatus").textContent = `${fmt(rows.length, 0)} ${t("available")}`; $("disaggTableCount").textContent = `${fmt(rows.length, 0)} ${t("available")}`; renderDisaggTable(); }
function renderDisaggDetail() { const row = state.disaggSelected; if (!row) { $("disaggDetail").innerHTML = `<i data-lucide="waves" aria-hidden="true"></i><p>${t("clickStation")}</p>`; $("disaggChart").innerHTML = ""; refreshIcons(); return; } const keys = STATION_DURATIONS; $("disaggDetail").innerHTML = `<div class="detail-header"><div><span class="eyebrow">${t("station")}</span><h2>${row.id}</h2><p class="detail-meta">${fmt(row.lat, 4)}, ${fmt(row.lon, 4)} · ${statusText(row.status)}</p></div><span class="status-tag">${fmt(row.years, 1)} ${t("years")}</span></div><p class="small-note">${t("fit")}: ${fmt(row.r2, 3)} · ${t("observations")}: ${fmt(row.nObs, 0)}</p>`; if (window.Plotly) Plotly.newPlot("disaggChart", [{x: keys.map(formatDurationKey), y: keys.map((key) => isFiniteNumber(row[key]) ? Number(row[key]) : null), type: "scatter", mode: "lines+markers", line: {color: "#176b78", width: 3}, marker: {color: "#e88b3a", size: 7}}], {margin: {l: 42, r: 8, t: 8, b: 38}, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {title: state.lang === "pt" ? "Duração" : "Duration"}, yaxis: {title: "Ratio", rangemode: "tozero"}, font: {family: "DM Sans", size: 9, color: "#16333f"}, showlegend: false}, {displayModeBar: false, responsive: true}); }
function renderDisaggTable() { const rows = getFilteredStations(state.disaggStatus || "all"); const shown = rows.slice(0, 700); $("disaggTable").innerHTML = shown.length ? `<table><thead><tr><th>${t("station")}</th><th>${t("latitude")}</th><th>${t("longitude")}</th><th>${t("status")}</th><th>5 min</th><th>30 min</th><th>1 h</th><th>24 h</th></tr></thead><tbody>${shown.map((row) => `<tr><td>${row.id}</td><td>${fmt(row.lat, 4)}</td><td>${fmt(row.lon, 4)}</td><td>${statusText(row.status)}</td><td class="table-number">${fmt(row.c5, 3)}</td><td class="table-number">${fmt(row.c30, 3)}</td><td class="table-number">${fmt(row.c60, 3)}</td><td class="table-number">${fmt(row.c1440, 3)}</td></tr>`).join("")}</tbody></table>` : `<p class="small-note">${t("tableEmpty")}</p>`; }
function renderStationSummary() { const rows = state.data.stations.records; const complete = rows.filter((row) => row.status !== "failed" && isFiniteNumber(row.c1440)); const accepted = rows.filter((row) => row.status === "ok-fit"); const support = rows.filter((row) => isFiniteNumber(row.years)); $("stationSummary").innerHTML = [[rows.length, t("stationTotal"), "#176b78"], [complete.length, t("stationComplete"), "#2d8c67"], [accepted.length, t("stationAccepted"), "#e88b3a"], [support.length, t("stationWithData"), "#cf5a53"]].map(([value, label, color]) => `<article class="summary-card"><span>${label}</span><strong style="color:${color}">${fmt(value, 0)}</strong><small>GRIDF-BR station export</small></article>`).join(""); renderStationTable(); }
function renderStationTable() { const rows = getFilteredStations(state.stationStatus || "all", $("stationSearch").value); const shown = rows.slice(0, 800); $("stationTable").innerHTML = shown.length ? `<table><thead><tr><th>${t("station")}</th><th>${t("latitude")}</th><th>${t("longitude")}</th><th>${t("status")}</th><th>${t("years")}</th><th>${t("observations")}</th><th>R²</th><th>5 min</th><th>30 min</th><th>1 h</th><th>24 h</th></tr></thead><tbody>${shown.map((row) => `<tr><td>${row.id}</td><td>${fmt(row.lat, 4)}</td><td>${fmt(row.lon, 4)}</td><td>${statusText(row.status)}</td><td>${fmt(row.years, 1)}</td><td>${fmt(row.nObs, 0)}</td><td>${fmt(row.r2, 3)}</td><td>${fmt(row.c5, 3)}</td><td>${fmt(row.c30, 3)}</td><td>${fmt(row.c60, 3)}</td><td>${fmt(row.c1440, 3)}</td></tr>`).join("")}</tbody></table>` : `<p class="small-note">${t("tableEmpty")}</p>`; $("stationTableFootnote").textContent = `${t("stationShown")} ${fmt(shown.length, 0)} ${t("stationOf")} ${fmt(rows.length, 0)} ${t("available")}.`; }
function renderMethods() { $("methodsGrid").innerHTML = `<article class="method-panel"><span class="eyebrow">${t("workflowTitle")}</span><h2>${t("workflowTitle")}</h2><p>${t("workflowText")}</p></article><article class="method-panel"><span class="eyebrow">${t("productTitle")}</span><h2>${t("productTitle")}</h2><p>${t("productText")}</p><ul>${PRODUCT_OPTIONS.map((product) => `<li><strong>${product[state.lang]}</strong></li>`).join("")}</ul></article><article class="method-panel"><span class="eyebrow">${t("stationTitle")}</span><h2>${t("stationTitle")}</h2><p>${t("stationText")}</p><h3>${t("limitsTitle")}</h3><p>${t("limitsText")}</p></article>`; $("figureGrid").innerHTML = FIGURES.map((figure) => `<article class="figure-card"><img src="${figure.file}" alt="${state.lang === "pt" ? figure.pt : figure.en}" loading="lazy"/><div class="figure-card-body"><h3>${state.lang === "pt" ? figure.pt : figure.en}</h3><p>${state.lang === "pt" ? figure.descPt : figure.descEn}</p></div></article>`).join(""); refreshIcons(); }
function setView(view) { state.view = view; document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view)); ["atlas", "disagg", "stations", "methods"].forEach((name) => { $(`${name}View`).hidden = name !== view; }); if (view === "disagg") { setTimeout(() => state.disaggMap?.invalidateSize(), 80); renderDisaggMap(); } if (view === "stations") renderStationSummary(); if (view === "methods") renderMethods(); if (view === "atlas") setTimeout(() => state.map?.invalidateSize(), 80); }
function setWelcomeStep(step) { state.welcomeStep = Math.max(0, Math.min(3, step)); document.querySelectorAll(".welcome-step").forEach((element) => { const active = Number(element.dataset.step) === state.welcomeStep; element.hidden = !active; element.classList.toggle("active", active); }); document.querySelectorAll(".welcome-dot").forEach((element) => element.classList.toggle("active", Number(element.dataset.step) === state.welcomeStep)); $("welcomeBack").disabled = state.welcomeStep === 0; $("welcomeNext").innerHTML = state.welcomeStep === 3 ? `<i data-lucide="check" aria-hidden="true"></i>${t("atlas")}` : `<span>${t("next")}</span><i data-lucide="arrow-right" aria-hidden="true"></i>`; refreshIcons(); }
function bindUi() { $("idfProduct").onchange = () => { state.product = $("idfProduct").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfMethod").onchange = () => { state.method = $("idfMethod").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfDuration").onchange = () => { state.duration = Number($("idfDuration").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfReturnPeriod").onchange = () => { state.returnPeriod = Number($("idfReturnPeriod").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfLayer").onchange = () => { state.layer = $("idfLayer").value; renderRaster(); }; $("disaggStatus").onchange = () => { state.disaggStatus = $("disaggStatus").value; renderDisaggMap(); }; $("disaggDuration").onchange = () => { state.disaggDuration = $("disaggDuration").value; renderDisaggDetail(); }; $("stationStatus").onchange = () => { state.stationStatus = $("stationStatus").value; renderStationTable(); }; $("stationSearch").oninput = renderStationTable; document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view))); document.querySelectorAll(".language-button").forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.language))); $("closeWelcomeButton").onclick = () => { $("welcomeModal").hidden = true; }; $("helpButton").onclick = () => { $("welcomeModal").hidden = false; setWelcomeStep(0); }; $("welcomeBack").onclick = () => setWelcomeStep(state.welcomeStep - 1); $("welcomeNext").onclick = () => state.welcomeStep === 3 ? ($("welcomeModal").hidden = true) : setWelcomeStep(state.welcomeStep + 1); document.querySelectorAll(".welcome-dot").forEach((button) => button.addEventListener("click", () => setWelcomeStep(Number(button.dataset.step)))); $("mobileControlsButton").onclick = () => $("atlasControls").classList.toggle("open"); $("downloadDisaggButton").onclick = () => downloadCsv("gridf_disaggregation_coefficients.csv", state.data.stations.records); $("downloadStationsButton").onclick = () => downloadCsv("gridf_high_resolution_stations.csv", getFilteredStations(state.stationStatus || "all", $("stationSearch").value)); $("downloadIdfGridButton").onclick = async () => { const entry = selectedCatalogEntry(); try { const response = await fetch(`${DATA_PATH}${entry.file.replace(/^.*?data\//, "")}`); downloadBlob(`gridf_${entry.product}_${entry.method}_BC_parameters.tif`, await response.blob()); } catch (error) { showToast(t("rasterError")); } }; }
function renderIdfEmpty() { $("idfDetail").innerHTML = `<div class="detail-empty"><div class="detail-icon"><i data-lucide="map-pin" aria-hidden="true"></i></div><span class="eyebrow">${t("selectedLocation")}</span><h2>${t("noLocation")}</h2><p>${t("clickMap")}</p><div class="detail-rule"></div><div class="detail-equation"><span class="eyebrow">${t("equation")}</span><strong>I = K · T<sup>a</sup> / (b + t)<sup>c</sup></strong><small>${t("equationNote")}</small></div></div>`; refreshIcons(); }
async function createApp() { try { state.data = {stations: await loadJson("stations.json"), brazil: await loadJson("brazil.geojson"), states: await loadJson("states.geojson"), idfCatalog: await loadJson("idf-catalog.json"), catalog: await loadJson("catalog.json")}; state.disaggStatus = "all"; state.stationStatus = "all"; state.disaggDuration = "c5"; populateSelects(); bindUi(); initAtlasMap(); initDisaggMap(); renderIdfEmpty(); renderMethods(); renderAtlasMeta(); setWelcomeStep(0); refreshIcons(); await renderRaster(); } catch (error) { console.error(error); showToast(error.message); } }

const REPORT_COPY = {
  en: {
    locationData: "Location and selected data",
    theory: "IDF theory and calculation",
    datasets: "Datasets and workflow",
    interpretation: "Interpretation and use",
    theoryText: "Intensity-duration-frequency relationships describe rainfall intensity as a function of storm duration and return period. GRIDF-BR uses the four-parameter Sherman relation I = K * RP^a / (b + t)^c, where I is intensity in mm/h, RP is the return period in years, t is duration in minutes, and K, a, b, and c are the gridded parameters sampled at the selected location.",
    datasetText: "This result uses the bias-corrected " ,
    workflowText: "Daily annual maxima are adjusted against the gauge calibration sample, converted to sub-daily durations using the selected disaggregation pathway, and represented by the archived Sherman parameter stack. Temporal-distribution coefficients are derived from ANA telemetric observations and interpolated with inverse-distance weighting for the gridded surfaces.",
    interpretationText: "The values are intended as a transparent spatial baseline and decision-support product. They should be interpreted with the grid resolution and calibration support in mind, and local engineering verification remains appropriate where site-specific observations are available.",
    parameters: "Sherman parameters"
  },
  pt: {
    locationData: "Local e dados selecionados",
    theory: "Teoria e calculo IDF",
    datasets: "Dados e fluxo de trabalho",
    interpretation: "Interpretacao e uso",
    theoryText: "As relacoes intensidade-duracao-frequencia descrevem a intensidade da chuva em funcao da duracao da tempestade e do periodo de retorno. O GRIDF-BR usa a relacao Sherman de quatro parametros I = K * RP^a / (b + t)^c, em que I e a intensidade em mm/h, RP e o periodo de retorno em anos, t e a duracao em minutos, e K, a, b e c sao os parametros gradeados amostrados no local selecionado.",
    datasetText: "Este resultado usa a pilha de parametros corrigida por vies do produto ",
    workflowText: "Os maximos anuais diarios sao ajustados em relacao a amostra de calibracao de gauges, convertidos para duracoes subdiarias pelo caminho de desagregacao selecionado e representados pela pilha Sherman arquivada. Os coeficientes de distribuicao temporal sao derivados de observacoes telemetricas da ANA e interpolados por ponderacao pelo inverso da distancia para as superficies gradeadas.",
    interpretationText: "Os valores sao destinados a uma linha de base espacial transparente e a um produto de apoio a decisao. Eles devem ser interpretados considerando a resolucao da grade e o suporte da calibracao; a verificacao de engenharia local continua apropriada quando houver observacoes especificas do local.",
    parameters: "Parametros Sherman"
  }
};
const reportText = (key) => REPORT_COPY[state.lang][key];

function selectedDisaggEntry() { return (state.data.disaggCatalog?.records || []).find((entry) => entry.family === state.disaggFamily && Number(entry.durationMin) === Number(state.disaggDuration)); }
async function loadDisaggRaster(entry) { const key = entry.file; if (state.disaggRasterCache.has(key)) return state.disaggRasterCache.get(key); if (!window.GeoTIFF) throw new Error("GeoTIFF library unavailable"); const response = await fetch(DATA_PATH + entry.file); if (!response.ok) throw new Error(entry.file + " (" + response.status + ")"); const tiff = await GeoTIFF.fromArrayBuffer(await response.arrayBuffer()); const image = await tiff.getImage(); const arrays = await image.readRasters({interleave: false}); const source = Array.isArray(arrays) ? arrays[0] : arrays; const raster = {entry, image, source, width: image.getWidth(), height: image.getHeight(), bounds: image.getBoundingBox()}; state.disaggRasterCache.set(key, raster); return raster; }
function disaggValues(raster) { const values = new Float32Array(raster.width * raster.height); values.fill(NaN); const nodata = Number(raster.entry.nodata); for (let i = 0; i < values.length; i += 1) { const value = Number(raster.source[i]); if (Number.isFinite(value) && value !== nodata) values[i] = value; } return values; }
function renderDisaggLegend(entry, scale) { $("disaggLegend").innerHTML = "<div class='disagg-legend-title'>" + (state.lang === "pt" ? entry.familyLabelPt : entry.familyLabel) + " · " + (state.lang === "pt" ? entry.durationLabelPt : entry.durationLabel) + "</div><div class='disagg-legend-bar'></div><div class='disagg-legend-labels'><span>" + fmt(scale.min, 2) + "</span><span>" + fmt(scale.max, 2) + "</span></div>"; }
async function renderDisaggSurface() { const token = ++state.disaggOverlayToken; const entry = selectedDisaggEntry(); if (!entry || !state.disaggMap) return; $("disaggMapStatus").textContent = t("rasterLoading"); try { const raster = await loadDisaggRaster(entry); if (token !== state.disaggOverlayToken) return; state.disaggRaster = raster; const values = disaggValues(raster); const min = quantile(values, .02); const max = quantile(values, .98); const canvas = document.createElement("canvas"); canvas.width = raster.width; canvas.height = raster.height; const context = canvas.getContext("2d"); const pixels = context.createImageData(raster.width, raster.height); for (let i = 0; i < values.length; i += 1) { const offset = i * 4; if (!Number.isFinite(values[i])) { pixels.data[offset] = 204; pixels.data[offset + 1] = 214; pixels.data[offset + 2] = 210; pixels.data[offset + 3] = 150; continue; } const rgb = colorFor(values[i], min, max); pixels.data[offset] = rgb[0]; pixels.data[offset + 1] = rgb[1]; pixels.data[offset + 2] = rgb[2]; pixels.data[offset + 3] = 205; } context.putImageData(pixels, 0, 0); maskCanvasToBrazil(context, raster.bounds, raster.width, raster.height); if (state.disaggOverlay) state.disaggOverlay.remove(); state.disaggOverlay = L.imageOverlay(canvas.toDataURL("image/png"), [[raster.bounds[1], raster.bounds[0]], [raster.bounds[3], raster.bounds[2]]], {opacity: .78, interactive: false}).addTo(state.disaggMap); state.stationLayer.eachLayer((layer) => layer.bringToFront?.()); $("disaggMapStatus").textContent = (state.lang === "pt" ? entry.durationLabelPt : entry.durationLabel) + " · " + fmt(entry.valid, 0) + " " + t("pixels"); $("disaggMapMeta").textContent = (state.lang === "pt" ? entry.familyLabelPt : entry.familyLabel) + " · IDW k=10, p=2"; renderDisaggLegend(entry, {min, max}); } catch (error) { console.error(error); $("disaggMapStatus").textContent = t("rasterError"); showToast(t("rasterError")); } }
function renderDisaggStationSummary(rows) { const complete = rows.filter((row) => row.status !== "failed" && isFiniteNumber(row.c1440)); const accepted = rows.filter((row) => row.status === "ok-fit"); $("disaggStationSummary").innerHTML = [[rows.length, t("stationTotal")], [complete.length, t("stationComplete")], [accepted.length, t("stationAccepted")]].map(([value, label]) => "<div class='station-summary-item'><span>" + label + "</span><strong>" + fmt(value, 0) + "</strong></div>").join(""); }
function renderDisaggTable() { const rows = getFilteredStations("all"); const rank = {"ok-fit": 0, "violated-fit": 1, failed: 2}; const shown = rows.slice().sort((a, b) => (rank[a.status] ?? 3) - (rank[b.status] ?? 3)).slice(0, 32); $("disaggTableCount").textContent = fmt(rows.length, 0) + " " + t("available"); $("disaggTable").innerHTML = shown.length ? "<table><thead><tr><th>" + t("station") + "</th><th>" + t("latitude") + "</th><th>" + t("longitude") + "</th><th>" + t("status") + "</th><th>5 min</th><th>30 min</th><th>1 h</th><th>24 h</th></tr></thead><tbody>" + shown.map((row) => "<tr><td>" + row.id + "</td><td>" + fmt(row.lat, 4) + "</td><td>" + fmt(row.lon, 4) + "</td><td>" + statusText(row.status) + "</td><td class='table-number'>" + fmt(row.c5, 3) + "</td><td class='table-number'>" + fmt(row.c30, 3) + "</td><td class='table-number'>" + fmt(row.c60, 3) + "</td><td class='table-number'>" + fmt(row.c1440, 3) + "</td></tr>").join("") + "</tbody></table>" : "<p class='small-note'>" + t("tableEmpty") + "</p>"; }
function initDisaggMap() { state.disaggMap = L.map("disaggMap", {zoomControl: false, preferCanvas: true}).setView([-14.2, -52.5], 4); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "&copy; OpenStreetMap contributors", maxZoom: 11}).addTo(state.disaggMap); L.control.zoom({position: "bottomleft"}).addTo(state.disaggMap); state.disaggBoundary = L.geoJSON(state.data.brazil, {style: {color: "#16333f", weight: 1.3, fillColor: "#75a9a0", fillOpacity: .08}}).addTo(state.disaggMap); state.stationLayer = L.layerGroup().addTo(state.disaggMap); state.disaggMap.fitBounds(state.disaggBoundary.getBounds(), {padding: [14, 14]}); $("disaggZoomInButton").onclick = () => state.disaggMap.zoomIn(); $("disaggZoomOutButton").onclick = () => state.disaggMap.zoomOut(); }
function renderDisaggMap() { if (!state.stationLayer) return; const rows = getFilteredStations("all"); state.stationLayer.clearLayers(); rows.forEach((row) => { if (!isFiniteNumber(row.lat) || !isFiniteNumber(row.lon)) return; const marker = L.circleMarker([row.lat, row.lon], {radius: 3.3, color: stationMarkerColor(row), weight: .8, fillColor: stationMarkerColor(row), fillOpacity: .75}); marker.bindTooltip(String(row.id) + " · " + statusText(row.status), {direction: "top", offset: [0, -3]}); marker.on("click", () => { state.disaggSelected = row; renderDisaggDetail(); }); marker.addTo(state.stationLayer); }); renderDisaggStationSummary(rows); renderDisaggTable(); renderDisaggSurface(); }
function renderDisaggDetail() { const row = state.disaggSelected; if (!row) { $("disaggDetail").innerHTML = "<i data-lucide='waves' aria-hidden='true'></i><p>" + t("clickStation") + "</p>"; $("disaggChart").innerHTML = ""; refreshIcons(); return; } const keys = STATION_DURATIONS; $("disaggDetail").innerHTML = "<div class='detail-header'><div><span class='eyebrow'>" + t("station") + "</span><h2>" + row.id + "</h2><p class='detail-meta'>" + fmt(row.lat, 4) + ", " + fmt(row.lon, 4) + " · " + statusText(row.status) + "</p></div><span class='status-tag'>" + fmt(row.years, 1) + " " + t("years") + "</span></div><p class='small-note'>" + t("fit") + ": " + fmt(row.r2, 3) + " · " + t("observations") + ": " + fmt(row.nObs, 0) + "</p>"; if (window.Plotly) Plotly.newPlot("disaggChart", [{x: keys.map(formatDurationKey), y: keys.map((key) => isFiniteNumber(row[key]) ? Number(row[key]) : null), type: "scatter", mode: "lines+markers", line: {color: "#176b78", width: 3}, marker: {color: "#e88b3a", size: 7}}], {margin: {l: 42, r: 8, t: 8, b: 38}, height: 260, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {title: state.lang === "pt" ? "Duração" : "Duration"}, yaxis: {title: "Ratio", rangemode: "tozero"}, font: {family: "DM Sans", size: 9, color: "#16333f"}, showlegend: false}, {displayModeBar: false, responsive: true}); }
function curveRows(params) { return DURATION_VALUES.map((duration) => { const row = {duration: formatDuration(duration), duration_min: duration}; RETURN_PERIODS.forEach((period) => { const intensity = idfIntensity(params, duration, period); row["rp_" + period + "_yr_mm_h"] = intensity; row["rp_" + period + "_yr_mm"] = intensity == null ? null : intensity * duration / 60; }); return row; }); }
function renderIdfDetail() { const selected = state.selected; if (!selected) return; const params = selected.params; const rows = curveRows(params); const entry = selected.entry; const selectedRows = [5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => { const intensity = idfIntensity(params, duration, state.returnPeriod); return [formatDuration(duration), fmt(intensity, 2), fmt(intensity == null ? null : intensity * duration / 60, 2)]; }); $("idfDetail").innerHTML = "<div class='detail-header'><div><span class='eyebrow'>" + t("selectedLocation") + "</span><h2>" + fmt(selected.lat, 4) + ", " + fmt(selected.lon, 4) + "</h2><p class='detail-meta'>" + selectedProductLabel() + " · " + selectedMethodLabel() + "</p></div><span class='status-tag'>" + t("biasLabel") + "</span></div><div class='detail-source'><strong>" + t("source") + "</strong><br/>" + entry.productLabel + " · " + entry.methodLabel + " · " + entry.width + " x " + entry.height + " · " + Math.abs(entry.transform[0]).toFixed(2) + " deg</div><h3 class='detail-heading'>" + t("parameters") + "</h3><div class='parameter-grid'>" + ["K", "a", "b", "c"].map((key) => "<div class='parameter-card'><span>" + key + "</span><strong>" + fmt(params[key], key === "K" ? 1 : 3) + "</strong></div>").join("") + "</div><div class='parameter-grid'>" + ["R2", "RMSE", "KS_p", "Nyears"].map((key) => "<div class='parameter-card'><span>" + (key === "KS_p" ? "KS p" : key) + "</span><strong>" + fmt(params[key], key === "Nyears" ? 0 : 3) + "</strong></div>").join("") + "</div><h3 class='detail-heading'>" + t("designIntensities") + "</h3><div id='idfCurve' class='chart-box'></div><p class='small-note'>" + t("selectedReturnPeriod") + ": " + state.returnPeriod + " yr · " + t("selectedDuration") + ": " + formatDuration(state.duration) + "</p><div class='table-wrap'><table><thead><tr><th>" + t("duration") + "</th><th>" + t("intensity") + " (" + t("mmHour") + ")</th><th>" + t("depth") + " (" + t("mm") + ")</th></tr></thead><tbody>" + selectedRows.map((row) => "<tr><td>" + row[0] + "</td><td class='table-number'>" + row[1] + "</td><td class='table-number'>" + row[2] + "</td></tr>").join("") + "</tbody></table></div><div class='detail-actions'><button class='action-button primary' id='downloadSelectedWord' type='button'><i data-lucide='file-text' aria-hidden='true'></i>" + t("downloadWord") + "</button><button class='action-button secondary' id='downloadSelectedCsv' type='button'><i data-lucide='download' aria-hidden='true'></i>" + t("downloadCsv") + "</button></div>"; $("downloadSelectedWord").onclick = () => { const reportRows = [[t("duration"), t("intensity") + " (" + t("mmHour") + ")", t("depth") + " (" + t("mm") + ")"]].concat(selectedRows); const metadata = selectedProductLabel() + " | " + selectedMethodLabel() + " | " + t("biasLabel") + " | " + fmt(selected.lat, 4) + ", " + fmt(selected.lon, 4); const sections = [{heading: reportText("locationData"), paragraphs: ["Product: " + selectedProductLabel(), "Disaggregation method: " + selectedMethodLabel(), "Coordinates: latitude " + fmt(selected.lat, 4) + ", longitude " + fmt(selected.lon, 4), "Return period: " + state.returnPeriod + " years; selected duration: " + formatDuration(state.duration)], bullets: []}, {heading: reportText("parameters"), paragraphs: ["K = " + fmt(params.K, 3) + "; a = " + fmt(params.a, 4) + "; b = " + fmt(params.b, 3) + "; c = " + fmt(params.c, 4)], bullets: []}, {heading: reportText("theory"), paragraphs: [reportText("theoryText")], bullets: []}, {heading: reportText("datasets"), paragraphs: [reportText("datasetText") + selectedProductLabel() + ". " + reportText("workflowText")], bullets: []}, {heading: reportText("interpretation"), paragraphs: [reportText("interpretationText")], bullets: []}]; downloadWordReport("gridf_idf_summary.docx", "GRIDF-BR IDF summary", metadata, sections, reportRows); }; $("downloadSelectedCsv").onclick = () => downloadCsv("gridf_idf_values.csv", rows.map((row) => ({latitude: selected.lat, longitude: selected.lon, product: entry.product, disaggregation: entry.method, bias: "BC", ...row}))); refreshIcons(); if (window.Plotly) { const series = RETURN_PERIODS.map((period) => ({x: rows.map((row) => row.duration_min), y: rows.map((row) => row["rp_" + period + "_yr_mm_h"]), type: "scatter", mode: "lines+markers", name: period + " yr", line: {width: 2.2}, marker: {size: 5}})); Plotly.newPlot("idfCurve", series, {margin: {l: 50, r: 8, t: 8, b: 72}, height: 300, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {type: "linear", title: state.lang === "pt" ? "Duracao (min)" : "Duration (min)", tickmode: "array", tickvals: durationTicks(), ticktext: durationTicks().map(formatDuration), tickangle: -40, automargin: true}, yaxis: {title: state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", rangemode: "tozero", automargin: true}, legend: {orientation: "h", y: 1.22}, font: {family: "DM Sans", size: 9, color: "#16333f"}}, {displayModeBar: false, responsive: true}); } }
function bindUi() { $("idfProduct").onchange = () => { state.product = $("idfProduct").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfMethod").onchange = () => { state.method = $("idfMethod").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfDuration").onchange = () => { state.duration = Number($("idfDuration").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfReturnPeriod").onchange = () => { state.returnPeriod = Number($("idfReturnPeriod").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfLayer").onchange = () => { state.layer = $("idfLayer").value; renderRaster(); }; $("disaggFamily").onchange = () => { state.disaggFamily = $("disaggFamily").value; populateSelects(); renderDisaggMap(); }; $("disaggDuration").onchange = () => { state.disaggDuration = Number($("disaggDuration").value); renderDisaggSurface(); }; $("stationStatus").onchange = () => { state.stationStatus = $("stationStatus").value; renderStationTable(); }; $("stationSearch").oninput = renderStationTable; document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view))); document.querySelectorAll(".language-button").forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.language))); $("closeWelcomeButton").onclick = () => { $("welcomeModal").hidden = true; }; $("helpButton").onclick = () => { $("welcomeModal").hidden = false; setWelcomeStep(0); }; $("welcomeBack").onclick = () => setWelcomeStep(state.welcomeStep - 1); $("welcomeNext").onclick = () => state.welcomeStep === 3 ? ($("welcomeModal").hidden = true) : setWelcomeStep(state.welcomeStep + 1); document.querySelectorAll(".welcome-dot").forEach((button) => button.addEventListener("click", () => setWelcomeStep(Number(button.dataset.step)))); $("mobileControlsButton").onclick = () => $("atlasControls").classList.toggle("open"); $("disaggZoomInButton").onclick = () => state.disaggMap.zoomIn(); $("disaggZoomOutButton").onclick = () => state.disaggMap.zoomOut(); $("downloadDisaggButton").onclick = () => { const entry = selectedDisaggEntry(); if (!entry) return; downloadCsv("gridf_disaggregation_" + state.disaggFamily + "_" + state.disaggDuration + "min.csv", state.data.stations.records); }; $("downloadStationsButton").onclick = () => downloadCsv("gridf_high_resolution_stations.csv", getFilteredStations(state.stationStatus || "all", $("stationSearch").value)); $("downloadIdfGridButton").onclick = async () => { const entry = selectedCatalogEntry(); try { const response = await fetch(DATA_PATH + entry.file); downloadBlob("gridf_" + entry.product + "_" + entry.method + "_BC_parameters.tif", await response.blob()); } catch (error) { showToast(t("rasterError")); } }; }
async function createApp() { try { state.data = {stations: await loadJson("stations.json"), brazil: await loadJson("brazil.geojson"), states: await loadJson("states.geojson"), idfCatalog: await loadJson("idf-catalog.json"), disaggCatalog: await loadJson("disagg-catalog.json"), catalog: await loadJson("catalog.json")}; state.stationStatus = "all"; populateSelects(); bindUi(); initAtlasMap(); initDisaggMap(); renderIdfEmpty(); renderMethods(); renderAtlasMeta(); renderDisaggDetail(); setWelcomeStep(0); refreshIcons(); await renderRaster(); renderDisaggMap(); } catch (error) { console.error(error); showToast(error.message); } }

function populateSelects() { fillSelect("idfProduct", PRODUCT_OPTIONS, state.product, state.lang); fillSelect("idfMethod", METHOD_OPTIONS, state.method, state.lang); fillSelect("idfLayer", LAYER_OPTIONS, state.layer, state.lang); fillSelect("idfDuration", DURATION_VALUES.map((value) => ({value, label: formatDuration(value)})), state.duration); fillSelect("idfReturnPeriod", RETURN_PERIODS.map((value) => ({value, label: value + " yr"})), state.returnPeriod); fillSelect("disaggFamily", DISAGG_FAMILY_OPTIONS, state.disaggFamily, state.lang); const disaggRecords = state.data.disaggCatalog?.records || []; const availableDurations = disaggRecords.filter((entry) => entry.family === state.disaggFamily).map((entry) => ({value: entry.durationMin, label: state.lang === "pt" ? entry.durationLabelPt : entry.durationLabel})); fillSelect("disaggDuration", availableDurations, state.disaggDuration); if (!availableDurations.some((item) => Number(item.value) === Number(state.disaggDuration)) && availableDurations.length) { state.disaggDuration = Number(availableDurations[0].value); $("disaggDuration").value = state.disaggDuration; } fillSelect("stationStatus", [{value: "all", label: t("statusAll")}, {value: "ok-fit", label: t("statusOk")}, {value: "violated-fit", label: t("statusViolated")}, {value: "failed", label: t("statusFailed")}], state.stationStatus || "all"); }
function setView(view) { state.view = view; document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view)); ["atlas", "disagg", "stations", "methods"].forEach((name) => { $(name + "View").hidden = name !== view; }); if (view === "disagg") { setTimeout(() => { state.disaggMap?.invalidateSize(); if (state.disaggBoundary) state.disaggMap.fitBounds(state.disaggBoundary.getBounds(), {padding: [14, 14]}); }, 120); renderDisaggMap(); } if (view === "stations") renderStationSummary(); if (view === "methods") renderMethods(); if (view === "atlas") setTimeout(() => state.map?.invalidateSize(), 80); }

Object.assign(COPY.en, {stationText: "The Disaggregation tab displays the interpolated coefficient surfaces and station observations used to support the duration ratios. The selected station support records can be downloaded as a CSV.", disaggTableNote: "A compact sample of station records is shown here; the full support export is available from the coefficient download."});
Object.assign(COPY.pt, {stationText: "A aba Desagregação exibe as superfícies de coeficientes interpoladas e as observações das estações usadas para apoiar as razões de duração. Os registros de suporte podem ser baixados em CSV.", disaggTableNote: "Uma amostra compacta dos registros de estação é mostrada aqui; o exportador completo está disponível no download dos coeficientes."});
Object.assign(COPY.en, {disaggTitle: "Interpolated disaggregation maps", disaggIntro: "Explore the spatial coefficient surfaces used to convert daily or reference-duration rainfall into the durations used by the 0.1° IDFs."});
Object.assign(COPY.pt, {disaggTitle: "Mapas interpolados de desagregação", disaggIntro: "Explore as superfícies espaciais de coeficientes usadas para converter a chuva diária ou de duração de referência nas durações das curvas IDF."});
Object.assign(COPY.en, {languageLabel: "LANGUAGE", equationTitle: "Core equations", equationIntro: "The interface applies four compact relationships to move from gridded daily extremes to the displayed IDF values.", biasEquationLabel: "Upper-tail multiplicative correction", ratioEquationLabel: "Temporal-distribution ratio", shermanEquationLabel: "Sherman IDF relation", depthEquationLabel: "Depth conversion"});
Object.assign(COPY.pt, {languageLabel: "IDIOMA", equationTitle: "Equações principais", equationIntro: "A interface aplica quatro relações compactas para passar dos extremos diários gradeados aos valores IDF exibidos.", biasEquationLabel: "Correção multiplicativa da cauda superior", ratioEquationLabel: "Razão de distribuição temporal", shermanEquationLabel: "Relação IDF de Sherman", depthEquationLabel: "Conversão de intensidade em lâmina"});
Object.assign(COPY.en, {productText: "The local browser bundle includes bias-corrected parameter stacks for BR-DWGD, IMERG, CHIRPS, and PERSIANN-CDR. The interface retains the three archived disaggregation pathways: local/interpolated, CETESB fixed ratios, and station-derived ratios."});
Object.assign(COPY.pt, {productText: "O pacote local inclui pilhas de parâmetros corrigidas por viés para BR-DWGD, IMERG, CHIRPS e PERSIANN-CDR. A interface mantém os três caminhos de desagregação arquivados: local/interpolado, razões fixas CETESB e derivado de estações."});
function durationTicks() { const width = $("idfCurve")?.clientWidth || 420; if (width < 700) return [5, 360, 1440]; if (width < 1000) return [5, 60, 360, 1440]; return [5, 10, 30, 60, 360, 720, 1440]; }
function drawBoundaryRing(context, ring, bounds, width, height) { if (!ring?.length) return; ring.forEach(([lon, lat], index) => { const x = (lon - bounds[0]) / (bounds[2] - bounds[0]) * width; const y = (bounds[3] - lat) / (bounds[3] - bounds[1]) * height; if (index === 0) context.moveTo(x, y); else context.lineTo(x, y); }); context.closePath(); }
function drawBoundaryGeometry(context, geometry, bounds, width, height) { if (!geometry) return; if (geometry.type === "Polygon") geometry.coordinates.forEach((ring) => drawBoundaryRing(context, ring, bounds, width, height)); if (geometry.type === "MultiPolygon") geometry.coordinates.forEach((polygon) => polygon.forEach((ring) => drawBoundaryRing(context, ring, bounds, width, height))); if (geometry.type === "GeometryCollection") geometry.geometries.forEach((item) => drawBoundaryGeometry(context, item, bounds, width, height)); }
function maskCanvasToBrazil(context, bounds, width, height) { const source = state.data.brazil; context.save(); context.globalCompositeOperation = "destination-in"; context.beginPath(); (source.features || [source]).forEach((feature) => drawBoundaryGeometry(context, feature.geometry || feature, bounds, width, height)); context.fill("evenodd"); context.restore(); }
function renderMethods() { $("methodsGrid").innerHTML = "<article class='method-panel'><span class='eyebrow'>" + t("workflowTitle") + "</span><h2>" + t("workflowTitle") + "</h2><p>" + t("workflowText") + "</p></article><article class='method-panel'><span class='eyebrow'>" + t("productTitle") + "</span><h2>" + t("productTitle") + "</h2><p>" + t("productText") + "</p><ul>" + PRODUCT_OPTIONS.map((product) => "<li><strong>" + product[state.lang] + "</strong></li>").join("") + "</ul></article><article class='method-panel'><span class='eyebrow'>" + t("stationTitle") + "</span><h2>" + t("stationTitle") + "</h2><p>" + t("stationText") + "</p><h3>" + t("limitsTitle") + "</h3><p>" + t("limitsText") + "</p></article><article class='method-panel'><span class='eyebrow'>" + t("referenceTitle") + "</span><h2>" + t("referenceTitle") + "</h2><p>" + t("referenceText") + "</p></article>"; $("figureGrid").innerHTML = FIGURES.map((figure) => "<article class='figure-card'><img src='" + figure.file + "' alt='" + (state.lang === "pt" ? figure.pt : figure.en) + "' loading='lazy'/><div class='figure-card-body'><h3>" + (state.lang === "pt" ? figure.pt : figure.en) + "</h3><p>" + (state.lang === "pt" ? figure.descPt : figure.descEn) + "</p></div></article>").join(""); refreshIcons(); }
function populateSelects() { fillSelect("idfProduct", PRODUCT_OPTIONS, state.product, state.lang); fillSelect("idfMethod", METHOD_OPTIONS, state.method, state.lang); fillSelect("idfLayer", LAYER_OPTIONS, state.layer, state.lang); fillSelect("idfDuration", DURATION_VALUES.map((value) => ({value, label: formatDuration(value)})), state.duration); fillSelect("idfReturnPeriod", RETURN_PERIODS.map((value) => ({value, label: value + " yr"})), state.returnPeriod); fillSelect("disaggFamily", DISAGG_FAMILY_OPTIONS, state.disaggFamily, state.lang); const disaggRecords = state.data.disaggCatalog?.records || []; const availableDurations = disaggRecords.filter((entry) => entry.family === state.disaggFamily).map((entry) => ({value: entry.durationMin, label: state.lang === "pt" ? entry.durationLabelPt : entry.durationLabel})); fillSelect("disaggDuration", availableDurations, state.disaggDuration); if (!availableDurations.some((item) => Number(item.value) === Number(state.disaggDuration)) && availableDurations.length) { state.disaggDuration = Number(availableDurations[0].value); $("disaggDuration").value = state.disaggDuration; } }
function setView(view) { state.view = view; document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view)); ["atlas", "disagg", "methods"].forEach((name) => { $(name + "View").hidden = name !== view; }); if (view === "disagg") { setTimeout(() => { state.disaggMap?.invalidateSize(); if (state.disaggBoundary) state.disaggMap.fitBounds(state.disaggBoundary.getBounds(), {padding: [14, 14]}); }, 120); renderDisaggMap(); } if (view === "methods") renderMethods(); if (view === "atlas") setTimeout(() => state.map?.invalidateSize(), 80); }
function bindUi() { $("idfProduct").onchange = () => { state.product = $("idfProduct").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfMethod").onchange = () => { state.method = $("idfMethod").value; state.selected = null; renderRaster(); renderIdfEmpty(); }; $("idfDuration").onchange = () => { state.duration = Number($("idfDuration").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfReturnPeriod").onchange = () => { state.returnPeriod = Number($("idfReturnPeriod").value); renderRaster(); if (state.selected) renderIdfDetail(); }; $("idfLayer").onchange = () => { state.layer = $("idfLayer").value; renderRaster(); }; $("disaggFamily").onchange = () => { state.disaggFamily = $("disaggFamily").value; populateSelects(); renderDisaggMap(); }; $("disaggDuration").onchange = () => { state.disaggDuration = Number($("disaggDuration").value); renderDisaggSurface(); }; document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view))); document.querySelectorAll(".language-button").forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.language))); $("closeWelcomeButton").onclick = () => { $("welcomeModal").hidden = true; }; $("helpButton").onclick = () => { $("welcomeModal").hidden = false; setWelcomeStep(0); }; $("welcomeBack").onclick = () => setWelcomeStep(state.welcomeStep - 1); $("welcomeNext").onclick = () => state.welcomeStep === 3 ? ($("welcomeModal").hidden = true) : setWelcomeStep(state.welcomeStep + 1); document.querySelectorAll(".welcome-dot").forEach((button) => button.addEventListener("click", () => setWelcomeStep(Number(button.dataset.step)))); $("mobileControlsButton").onclick = () => $("atlasControls").classList.toggle("open"); $("disaggZoomInButton").onclick = () => state.disaggMap.zoomIn(); $("disaggZoomOutButton").onclick = () => state.disaggMap.zoomOut(); $("downloadDisaggButton").onclick = () => { const entry = selectedDisaggEntry(); if (!entry) return; downloadCsv("gridf_disaggregation_" + state.disaggFamily + "_" + state.disaggDuration + "min.csv", state.data.stations.records); }; $("downloadIdfGridButton").onclick = async () => { const entry = selectedCatalogEntry(); try { const response = await fetch(DATA_PATH + entry.file); downloadBlob("gridf_" + entry.product + "_" + entry.method + "_BC_parameters.tif", await response.blob()); } catch (error) { showToast(t("rasterError")); } }; }
async function createApp() { try { state.data = {stations: await loadJson("stations.json"), brazil: await loadJson("brazil.geojson"), states: await loadJson("states.geojson"), idfCatalog: await loadJson("idf-catalog.json"), disaggCatalog: await loadJson("disagg-catalog.json"), catalog: await loadJson("catalog.json")}; populateSelects(); bindUi(); initAtlasMap(); initDisaggMap(); renderIdfEmpty(); renderMethods(); renderAtlasMeta(); renderDisaggDetail(); setWelcomeStep(0); refreshIcons(); await renderRaster(); renderDisaggMap(); } catch (error) { console.error(error); showToast(error.message); } }

function renderMethods() {
  const equationPanel = "<article class='method-panel equation-panel'><span class='eyebrow'>" + t("equationTitle") + "</span><h2>" + t("equationTitle") + "</h2><p>" + t("equationIntro") + "</p><div class='equation-list'><div class='equation-item'><strong>P<sub>corr</sub> = λ · P<sub>grid</sub></strong><span>" + t("biasEquationLabel") + "</span></div><div class='equation-item'><strong>C<sub>d</sub> = P<sub>d</sub> / P<sub>ref</sub></strong><span>" + t("ratioEquationLabel") + "</span></div><div class='equation-item'><strong>I = K · RP<sup>a</sup> / (b + t)<sup>c</sup></strong><span>" + t("shermanEquationLabel") + "</span></div><div class='equation-item'><strong>D = I · t / 60</strong><span>" + t("depthEquationLabel") + "</span></div></div></article>";
  $("methodsGrid").innerHTML = "<article class='method-panel'><span class='eyebrow'>" + t("workflowTitle") + "</span><h2>" + t("workflowTitle") + "</h2><p>" + t("workflowText") + "</p></article><article class='method-panel'><span class='eyebrow'>" + t("productTitle") + "</span><h2>" + t("productTitle") + "</h2><p>" + t("productText") + "</p><ul>" + PRODUCT_OPTIONS.map((product) => "<li><strong>" + product[state.lang] + "</strong></li>").join("") + "</ul></article>" + equationPanel + "<article class='method-panel'><span class='eyebrow'>" + t("stationTitle") + "</span><h2>" + t("stationTitle") + "</h2><p>" + t("stationText") + "</p><h3>" + t("limitsTitle") + "</h3><p>" + t("limitsText") + "</p></article>";
  $("figureGrid").innerHTML = FIGURES.map((figure) => "<article class='figure-card'><img src='" + figure.file + "' alt='" + (state.lang === "pt" ? figure.pt : figure.en) + "' loading='lazy'/><div class='figure-card-body'><h3>" + (state.lang === "pt" ? figure.pt : figure.en) + "</h3><p>" + (state.lang === "pt" ? figure.descPt : figure.descEn) + "</p></div></article>").join("");
  refreshIcons();
}
function openFigureLightbox(image) { const modal = $("figureLightbox"); const preview = $("figureLightboxImage"); const title = $("figureLightboxTitle"); if (!modal || !preview) return; preview.src = image.currentSrc || image.src; preview.alt = image.alt || ""; title.textContent = image.alt || ""; modal.hidden = false; $("closeFigureButton")?.focus(); }
function closeFigureLightbox() { const modal = $("figureLightbox"); if (modal) modal.hidden = true; }
document.addEventListener("click", (event) => { const image = event.target.closest?.(".figure-card img"); if (image) { openFigureLightbox(image); return; } if (event.target === $("figureLightbox") || event.target.closest?.("#closeFigureButton")) closeFigureLightbox(); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeFigureLightbox(); });
function patchIdfPlotLayout() { if (!window.Plotly || window.Plotly.__gridfIdfPatch) return; const basePlot = window.Plotly.newPlot.bind(window.Plotly); window.Plotly.newPlot = (target, data, layout, config) => { const targetId = typeof target === "string" ? target : target?.id; if (targetId === "idfCurve") { layout = {...layout, xaxis: {...layout?.xaxis, tickmode: "array", tickvals: durationTicks(), ticktext: durationTicks().map(formatDuration), tickangle: 0, title: state.lang === "pt" ? "Duração (min)" : "Duration (min)"}}; } return basePlot(target, data, layout, config); }; window.Plotly.__gridfIdfPatch = true; }
patchIdfPlotLayout();
setLanguage(state.lang);
createApp().then(() => setLanguage(state.lang));

Object.assign(COPY.en, {
  navCities: "Municipal IDFs",
  citiesKicker: "MUNICIPAL SCALE",
  citiesTitle: "Municipal 0.1° IDFs",
  citiesIntro: "Browse city-scale summaries of the bias-corrected gridded IDF parameters and the interpolated temporal-distribution coefficients.",
  citySearchLabel: "Search municipality",
  citySearchPlaceholder: "Name or code",
  cityStateLabel: "State",
  cityStateAll: "All states",
  cityMethodNoteTitle: "CITY SUMMARY",
  cityMethodNote: "Corrected annual maxima are area-weighted over each municipality, then Gumbel return levels are refitted by the method of moments and the Sherman relation is refitted at municipal scale.",
  cityDailyDepth: "Daily return depth",
  cityCoverage: "Mean valid area",
  cityYears: "Years used",
  cityScaleNote: "map scale: 5th–95th percentile",
  cityMapNote: "Click a municipality to inspect its IDF curve and coefficients.",
  cityMapHelp: "Select a municipality on the map.",
  selectedMunicipality: "SELECTED MUNICIPALITY",
  noMunicipality: "No municipality selected",
  clickMunicipality: "Click a municipality on the map.",
  municipalityRecords: "MUNICIPAL RECORDS",
  municipalityTableTitle: "City-scale IDF and coefficient inventory",
  cityTableNote: "The table follows the current filters. Select a row to open its city-scale IDF summary.",
  cityScale: "City-scale summary",
  cityCoefficient: "Temporal coefficient",
  cityDisaggTitle: "Disaggregation coefficients",
  cityDisaggNote: "The interpolated temporal ratios are area-weighted over the municipality and are used to refit the municipal IDF relation.",
  cityDailyCoefficient: "Relative to daily maximum",
  cityReferenceCoefficient: "Relative to reference duration",
  cityIntensity: "Selected intensity",
  cityArea: "Municipal area",
  citySupport: "Valid / touched cells",
  cityNoData: "The selected product does not have enough valid annual or interpolated-coefficient support to produce a municipal fit.",
  cityAllStates: "All states",
  cityDownloadWord: "Word summary",
  cityDownloadCsv: "CSV values",
  citySourceNote: "Municipal annual maxima were aggregated from corrected raster series using polygon-cell intersection areas; Gumbel return levels were fitted by the method of moments and Sherman parameters were then refitted.",
  cityMapStatus: "Municipalities"
});
Object.assign(COPY.pt, {
  navCities: "IDFs municipais",
  citiesKicker: "ESCALA MUNICIPAL",
  citiesTitle: "IDFs municipais",
  citiesIntro: "Consulte resumos em escala municipal dos parâmetros IDF gradeados corrigidos por viés e dos coeficientes interpolados de distribuição temporal.",
  citySearchLabel: "Buscar município",
  citySearchPlaceholder: "Nome ou código",
  cityStateLabel: "Estado",
  cityStateAll: "Todos os estados",
  cityMethodNoteTitle: "RESUMO MUNICIPAL",
  cityMethodNote: "Os máximos anuais corrigidos são ponderados pela área de cada município; os níveis de retorno Gumbel são reajustados pelo método dos momentos e a relação Sherman é reajustada na escala municipal.",
  cityDailyDepth: "Lâmina diária de retorno",
  cityCoverage: "Área média válida",
  cityYears: "Anos utilizados",
  cityScaleNote: "escala do mapa: percentis 5–95",
  cityMapNote: "Clique em um município para inspecionar sua curva IDF e seus coeficientes.",
  cityMapHelp: "Selecione um município no mapa.",
  selectedMunicipality: "MUNICÍPIO SELECIONADO",
  noMunicipality: "Nenhum município selecionado",
  clickMunicipality: "Clique em um município no mapa.",
  municipalityRecords: "REGISTROS MUNICIPAIS",
  municipalityTableTitle: "Inventário municipal de IDF e coeficientes",
  cityTableNote: "A tabela acompanha os filtros atuais. Selecione uma linha para abrir o resumo municipal de IDF.",
  cityScale: "Resumo em escala municipal",
  cityCoefficient: "Coeficiente temporal",
  cityDisaggTitle: "Coeficientes de desagregação",
  cityDisaggNote: "As razões temporais interpoladas são ponderadas pela área do município e usadas para reajustar a relação IDF municipal.",
  cityDailyCoefficient: "Relativo ao máximo diário",
  cityReferenceCoefficient: "Relativo à duração de referência",
  cityIntensity: "Intensidade selecionada",
  cityArea: "Área municipal",
  citySupport: "Células válidas / intersectadas",
  cityNoData: "O produto selecionado não tem suporte anual ou de coeficientes interpolados suficiente para produzir um ajuste municipal.",
  cityAllStates: "Todos os estados",
  cityDownloadWord: "Resumo Word",
  cityDownloadCsv: "Valores CSV",
  citySourceNote: "Os máximos anuais municipais foram agregados das grades corrigidas usando as áreas de interseção entre polígonos e células; os níveis de retorno Gumbel foram ajustados pelo método dos momentos e os parâmetros Sherman foram então reajustados.",
  cityMapStatus: "Municípios"
});

Object.assign(COPY.en, {
  downloadCityCatalog: "Download complete municipal catalogue",
  cityCatalogDownloadNote: "Exports all municipalities for the selected product, duration, and return period. Municipal frequency analysis uses Gumbel fitted by the method of moments.",
  cityReportTitle: "Municipal IDF report",
  cityReportSubtitle: "Bias-corrected gridded IDF summary and interpolated temporal-distribution coefficient",
  cityReportMethod: "Method and data",
  cityReportTheory: "The IDF relation expresses rainfall intensity as a function of storm duration and return period. GRIDF-BR uses I = K · RP^a / (b + t)^c, where I is intensity in mm/h, RP is return period in years, t is duration in minutes, and K, a, b, and c are the parameters sampled from the selected gridded stack.",
  cityReportMunicipal: "The corrected annual-max raster values are aggregated for the municipality using polygon-cell intersection areas. The resulting municipal series is a first-order areal approximation because the archived input is an annual-max grid rather than daily fields from which a municipal annual maximum could be formed directly.",
  cityReportWorkflow: "The selected product is fitted with a Gumbel distribution by the method of moments using the available municipal annual values. Return depths are converted to sub-daily durations with the area-weighted local/interpolated temporal ratios, and the four-parameter Sherman relation is refitted to those municipal intensities. The report provides both interpolated coefficient families for transparency.",
  cityReportInterpretation: "The report is intended for transparent assessment and decision support. Values should be interpreted with the native grid resolution, spatial averaging, calibration support, and the limitations of the underlying frequency analysis in mind. Local engineering verification remains appropriate where site-specific observations are available.",
  cityReportLocation: "Municipality and selected configuration",
  cityReportParameters: "Municipal parameters",
  cityReportFigureMap: "Municipal location within the Brazil grid",
  cityReportFigureCurve: "Intensity-duration-frequency curves",
  cityReportDesignTable: "Design intensities at the selected return period",
  cityReportDisagg: "Municipal disaggregation coefficients",
  cityReportDisaggNote: "The two coefficient pathways are reported for every duration available in the interpolated surfaces. Their municipal values are area-weighted with the same polygon-cell intersection approach used for the annual-max series.",
  cityReportFigureNote: "Figures are generated locally at a resolution equivalent to 600 dpi for the report layout.",
  cityReportGenerated: "Generated by the GRIDF-BR municipal IDF tool"
});
Object.assign(COPY.pt, {
  downloadCityCatalog: "Baixar catálogo municipal completo",
  cityCatalogDownloadNote: "Exporta todos os municípios para o produto, duração e período de retorno selecionados. A análise de frequência municipal usa Gumbel ajustado pelo método dos momentos.",
  cityReportTitle: "Relatório municipal de IDF",
  cityReportSubtitle: "Resumo IDF gradeado corrigido por viés e coeficiente interpolado de distribuição temporal",
  cityReportMethod: "Método e dados",
  cityReportTheory: "A relação IDF expressa a intensidade da chuva em função da duração da tempestade e do período de retorno. Na toolbox do GRIDF-BR, a relação é representada por I = K · RP^a / (b + t)^c, em que I é a intensidade em mm/h, RP é o período de retorno em anos, t é a duração em minutos e K, a, b e c são os parâmetros amostrados da grade selecionada.",
  cityReportMunicipal: "Os valores corrigidos dos máximos anuais na grade são agregados para o município usando as áreas de interseção entre polígonos e células. A série municipal resultante é uma aproximação areal de primeira ordem, pois o arquivo disponível é uma grade de máximos anuais, e não campos diários a partir dos quais se possa formar diretamente o máximo anual da média municipal.",
  cityReportWorkflow: "O produto selecionado é ajustado pela distribuição Gumbel, pelo método dos momentos, usando os valores anuais municipais disponíveis. As lâminas de retorno são convertidas para durações subdiárias com as razões temporais locais/interpoladas ponderadas pela área, e a relação Sherman de quatro parâmetros é reajustada para essas intensidades municipais. O relatório apresenta as duas famílias de coeficientes interpolados para transparência.",
  cityReportInterpretation: "O relatório foi preparado para apoiar avaliações transparentes e decisões preliminares. Os valores devem ser interpretados considerando a resolução nativa da grade, a média espacial, o suporte da calibração e as limitações da análise de frequência subjacente.",
  cityReportLocation: "Município e configuração selecionada",
  cityReportParameters: "Parâmetros municipais",
  cityReportFigureMap: "Localização municipal na grade do Brasil",
  cityReportFigureCurve: "Curvas intensidade-duração-frequência",
  cityReportDesignTable: "Intensidades de projeto no período de retorno selecionado",
  cityReportDisagg: "Coeficientes municipais de desagregação",
  cityReportDisaggNote: "As duas famílias de coeficientes são apresentadas para cada duração disponível nas superfícies interpoladas. Os valores municipais são ponderados pela área usando a mesma abordagem de interseção entre polígonos e células aplicada à série de máximos anuais.",
  cityReportFigureNote: "As figuras são geradas localmente em resolução equivalente a 600 dpi para o layout do relatório.",
  cityReportGenerated: "Gerado pela ferramenta municipal de IDF do GRIDF-BR"
});

Object.assign(state, {cityMap: null, cityLayer: null, citySelected: null, citySearch: "", cityState: "all", cityProduct: state.product, cityMethod: "local-interpolated", cityDuration: 60, cityReturnPeriod: 10, cityPlotReturnPeriods: DEFAULT_PLOT_RETURN_PERIODS.slice()});

function cityRecords() { return state.data.cityCatalog?.records || []; }
function cityRecord(code) { return cityRecords().find((record) => String(record.code) === String(code)); }
function cityText(value) { return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;"); }
function cityProductLabel() { return PRODUCT_OPTIONS.find((item) => item.value === state.cityProduct)?.[state.lang] || state.cityProduct; }
function cityMethodLabel() { return CITY_METHOD_OPTIONS.find((item) => item.value === state.cityMethod)?.[state.lang] || state.cityMethod; }
function cityKey() { return `${state.cityProduct}|${state.cityMethod}`; }
function cityParams(record) { return record?.idf?.[cityKey()] || null; }
function cityCoefficient(record) { return record?.disagg?.relative_to_daily?.[String(state.cityDuration)] ?? null; }
function cityReferenceCoefficient(record) { return record?.disagg?.relative_to_subdaily?.[String(state.cityDuration)] ?? null; }
function cityDisaggRows(record) {
  const daily = record?.disagg?.relative_to_daily || {};
  const reference = record?.disagg?.relative_to_subdaily || {};
  const durations = [...new Set([...Object.keys(daily), ...Object.keys(reference)].map(Number).filter(Number.isFinite))].sort((left, right) => left - right);
  return durations.map((duration) => ({
    duration,
    daily: daily[String(duration)] ?? null,
    reference: reference[String(duration)] ?? null
  }));
}
function cityIntensity(record) { const params = cityParams(record); return params ? idfIntensity(params, state.cityDuration, state.cityReturnPeriod) : null; }
function filteredCityRecords() {
  const query = String(state.citySearch || "").trim().toLocaleLowerCase();
  return cityRecords().filter((record) => {
    const matchesState = state.cityState === "all" || record.stateCode === state.cityState;
    const haystack = `${record.name} ${record.code} ${record.state}`.toLocaleLowerCase();
    return matchesState && (!query || haystack.includes(query));
  }).sort((left, right) => left.name.localeCompare(right.name, state.lang === "pt" ? "pt-BR" : "en-US"));
}
function cityColor(value, min, max) { const rgb = colorFor(value, min, max); return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`; }
function cityScale(records) { const values = records.map(cityIntensity).filter(isFiniteNumber).sort((left, right) => left - right); if (!values.length) return {min: 0, max: 1, count: 0, hasData: false}; let min = values[Math.floor((values.length - 1) * .05)]; let max = values[Math.floor((values.length - 1) * .95)]; if (min === max) { const padding = Math.max(Math.abs(min) * .05, .01); min -= padding; max += padding; } return {min, max, count: values.length, hasData: true}; }
function downloadCityCatalog() {
  const rows = [];
  cityRecords().forEach((record) => PRODUCT_OPTIONS.forEach((product) => CITY_METHOD_OPTIONS.forEach((method) => {
    const params = record.idf?.[product.value + "|" + method.value] || {};
    rows.push({
      municipality: record.name,
      municipality_code: record.code,
      state: record.state,
      state_code: record.stateCode,
      latitude: record.latitude,
      longitude: record.longitude,
      area_km2: record.areaKm2,
      product: product[state.lang],
      disaggregation: method[state.lang],
      source_product: params.sourceProduct,
      duration_min: state.cityDuration,
      return_period_years: state.cityReturnPeriod,
      K: params.K,
      a: params.a,
      b: params.b,
      c: params.c,
      temporal_coefficient_relative_to_daily: record.disagg?.relative_to_daily?.[String(state.cityDuration)] ?? null,
      temporal_coefficient_relative_to_reference_duration: record.disagg?.relative_to_subdaily?.[String(state.cityDuration)] ?? null,
      intensity_mm_h: params.K == null ? null : idfIntensity(params, state.cityDuration, state.cityReturnPeriod),
      R2: params.R2,
      RMSE: params.RMSE,
      KS_p: params.KS_p,
      Nyears: params.Nyears,
      valid_grid_cells: params.validPixels,
      touched_grid_cells: params.touchedCells,
      valid_area_fraction_mean: params.validAreaFractionMean,
      valid_area_fraction_min: params.validAreaFractionMin,
      valid_area_fraction_max: params.validAreaFractionMax,
      daily_return_depth_mm: params.q24?.[String(state.cityReturnPeriod)] ?? null
    });
  })));
  downloadCsv("gridf_municipal_catalog_complete.csv", rows);
}

function initCityMap() {
  state.cityMap = L.map("cityMap", {zoomControl: false, preferCanvas: true}).setView([-14.2, -52.5], 4);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {attribution: "&copy; OpenStreetMap contributors", maxZoom: 11}).addTo(state.cityMap);
  L.control.zoom({position: "bottomleft"}).addTo(state.cityMap);
  state.cityLayer = L.geoJSON(state.data.cities, {
    style: {color: "#557c7b", weight: .45, fillColor: "#69a99c", fillOpacity: .32},
    onEachFeature: (feature, layer) => {
      const record = cityRecord(feature.properties.code);
      if (record) layer.bindTooltip(`${record.name}, ${record.state}`, {direction: "top", sticky: true, opacity: .92});
      layer.on("click", () => selectCity(feature.properties.code));
      layer.on("mouseover", () => layer.setStyle({weight: 1.2, color: "#16333f"}));
      layer.on("mouseout", () => renderCityMap());
    }
  }).addTo(state.cityMap);
  state.cityMap.fitBounds(state.cityLayer.getBounds(), {padding: [14, 14]});
  $("cityZoomInButton").onclick = () => state.cityMap.zoomIn();
  $("cityZoomOutButton").onclick = () => state.cityMap.zoomOut();
}

function renderCityMap() {
  if (!state.cityLayer) return;
  const records = filteredCityRecords();
  const visible = new Set(records.map((record) => String(record.code)));
  const scale = cityScale(records);
  state.cityLayer.eachLayer((layer) => {
    const code = String(layer.feature?.properties?.code || "");
    const record = cityRecord(code);
    const value = record ? cityIntensity(record) : null;
    const isVisible = visible.has(code);
    const isSelected = state.citySelected?.code === code;
    layer.setStyle({
      color: isSelected ? "#16333f" : isVisible ? "#557c7b" : "#b7c8c3",
      weight: isSelected ? 2.2 : .45,
      fillColor: isFiniteNumber(value) ? cityColor(value, scale.min, scale.max) : "#c8d5d1",
      fillOpacity: isVisible ? (isSelected ? .86 : .48) : .04,
      opacity: isVisible ? .8 : .12
    });
  });
  renderCityLegend(scale);
  $("cityStatus").textContent = `${fmt(scale.count, 0)} / ${fmt(records.length, 0)} ${t("available")}`;
  $("cityMapTitle").textContent = `${cityProductLabel()} · Gumbel`;
  $("cityMapMeta").textContent = `${t("cityIntensity")}: ${formatDuration(state.cityDuration)} · ${state.cityReturnPeriod} yr`;
}

function renderCityLegend(scale) {
  const legend = $("cityLegend");
  if (!legend) return;
  legend.innerHTML = `<div class="legend-title">${t("cityIntensity")} · Gumbel · ${formatDuration(state.cityDuration)} · ${state.cityReturnPeriod} yr</div><div class="legend-bar"></div>${scale.hasData ? `<div class="legend-labels"><span>${fmt(scale.min, 1)} ${t("mmHour")}</span><span>${fmt(scale.max, 1)} ${t("mmHour")}</span></div><div class="legend-note">${fmt(scale.count, 0)} ${t("available")} · ${t("cityScaleNote")}</div>` : `<div class="legend-note">${t("cityNoData")}</div>`}`;
}

function renderCityTable() {
  const records = filteredCityRecords();
  const shown = records.slice(0, 1000);
  $("cityTableCount").textContent = `${fmt(records.length, 0)} ${t("available")}`;
  if (!shown.length) { $("cityTable").innerHTML = `<p class="small-note">${t("cityNoData")}</p>`; return; }
  $("cityTable").innerHTML = `<table><thead><tr><th>${t("municipalityRecords")}</th><th>${t("cityStateLabel")}</th><th>K</th><th>a</th><th>${t("cityCoefficient")}</th><th>${t("cityIntensity")} (${t("mmHour")})</th><th>R²</th></tr></thead><tbody>${shown.map((record) => { const params = cityParams(record) || {}; return `<tr data-city-code="${cityText(record.code)}" class="${state.citySelected?.code === record.code ? "selected" : ""}"><td class="municipality-name">${cityText(record.name)}</td><td>${cityText(record.state)}</td><td class="table-number">${fmt(params.K, 1)}</td><td class="table-number">${fmt(params.a, 3)}</td><td class="table-number city-coefficient">${fmt(cityCoefficient(record), 3)}</td><td class="table-number">${fmt(cityIntensity(record), 1)}</td><td class="table-number">${fmt(params.R2, 3)}</td></tr>`; }).join("")}</tbody></table>`;
  $("cityTable").querySelectorAll("tbody tr").forEach((row) => row.addEventListener("click", () => selectCity(row.dataset.cityCode)));
}

function renderCityDetail() {
  const record = state.citySelected;
  const panel = $("cityDetail");
  if (!record) {
    panel.innerHTML = `<div class="detail-empty"><div class="detail-icon"><i data-lucide="landmark" aria-hidden="true"></i></div><span class="eyebrow">${t("selectedMunicipality")}</span><h2>${t("noMunicipality")}</h2><p>${t("clickMunicipality")}</p></div>`;
    refreshIcons();
    return;
  }
  const params = cityParams(record);
  if (!params || !isFiniteNumber(params.K)) {
    panel.innerHTML = `<div class="detail-empty"><div class="detail-icon"><i data-lucide="landmark" aria-hidden="true"></i></div><span class="eyebrow">${t("selectedMunicipality")}</span><h2>${cityText(record.name)}</h2><p>${t("cityNoData")}</p></div>`;
    refreshIcons();
    return;
  }
  const rows = curveRows(params);
  const disaggRows = cityDisaggRows(record);
  const selectedRows = [5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => {
    const intensity = idfIntensity(params, duration, state.cityReturnPeriod);
    return [formatDuration(duration), fmt(intensity, 2), fmt(intensity == null ? null : intensity * duration / 60, 2)];
  });
  const support = `${fmt(params.validPixels, 0)} / ${fmt(params.touchedCells, 0)}`;
  const coverage = isFiniteNumber(params.validAreaFractionMean) ? `${fmt(Number(params.validAreaFractionMean) * 100, 1)}%` : t("noData");
  const dailyDepth = params.q24?.[String(state.cityReturnPeriod)] ?? null;
  panel.innerHTML = `<div class="detail-header"><div><span class="eyebrow">${t("selectedMunicipality")}</span><h2>${cityText(record.name)}</h2><p class="detail-meta">${cityText(record.state)} · ${fmt(record.latitude, 4)}, ${fmt(record.longitude, 4)}</p></div><span class="status-tag">${t("cityScale")}</span></div><div class="detail-source"><strong>${t("source")}</strong><br/>${t("citySourceNote")}<br/>${cityProductLabel()} · ${cityMethodLabel()} · Gumbel</div><h3 class="detail-heading">${t("parameters")}</h3><div class="parameter-grid">${["K", "a", "b", "c"].map((key) => `<div class="parameter-card"><span>${key}</span><strong>${fmt(params[key], key === "K" ? 1 : 3)}</strong></div>`).join("")}</div><div class="parameter-grid"><div class="parameter-card"><span>${t("cityDailyDepth")}</span><strong>${fmt(dailyDepth, 1)} mm</strong></div><div class="parameter-card"><span>${t("cityYears")}</span><strong>${fmt(params.Nyears, 0)}</strong></div><div class="parameter-card"><span>${t("cityCoverage")}</span><strong>${coverage}</strong></div><div class="parameter-card"><span>${t("citySupport")}</span><strong>${support}</strong></div></div><div class="parameter-grid"><div class="parameter-card"><span>${t("cityDailyCoefficient")}</span><strong>${fmt(cityCoefficient(record), 3)}</strong></div><div class="parameter-card"><span>${t("cityReferenceCoefficient")}</span><strong>${fmt(cityReferenceCoefficient(record), 3)}</strong></div><div class="parameter-card"><span>${t("cityArea")}</span><strong>${fmt(record.areaKm2, 0)} km²</strong></div><div class="parameter-card"><span>R²</span><strong>${fmt(params.R2, 3)}</strong></div></div><h3 class="detail-heading">${t("designIntensities")}</h3><div id="cityIdfCurve" class="chart-box"></div><p class="small-note">${t("selectedReturnPeriod")}: ${state.cityReturnPeriod} yr · ${t("selectedDuration")}: ${formatDuration(state.cityDuration)}</p><div class="table-wrap"><table><thead><tr><th>${t("duration")}</th><th>${t("intensity")} (${t("mmHour")})</th><th>${t("depth")} (${t("mm")})</th></tr></thead><tbody>${selectedRows.map((row) => `<tr><td>${row[0]}</td><td class="table-number">${row[1]}</td><td class="table-number">${row[2]}</td></tr>`).join("")}</tbody></table></div><h3 class="detail-heading">${t("cityDisaggTitle")}</h3><p class="small-note city-disagg-note">${t("cityDisaggNote")}</p><div id="cityDisaggChart" class="chart-box city-disagg-chart"></div><div class="table-wrap city-disagg-table"><table><thead><tr><th>${t("duration")}</th><th>${t("cityDailyCoefficient")}</th><th>${t("cityReferenceCoefficient")}</th></tr></thead><tbody>${disaggRows.map((row) => `<tr class="${row.duration === state.cityDuration ? "selected-duration" : ""}"><td>${formatDuration(row.duration)}</td><td class="table-number">${fmt(row.daily, 3)}</td><td class="table-number">${fmt(row.reference, 3)}</td></tr>`).join("")}</tbody></table></div><div class="detail-actions"><button class="action-button primary" id="downloadCityWord" type="button"><i data-lucide="file-text" aria-hidden="true"></i>${t("cityDownloadWord")}</button><button class="action-button secondary" id="downloadCityCsv" type="button"><i data-lucide="download" aria-hidden="true"></i>${t("cityDownloadCsv")}</button></div>`;
  $("downloadCityWord").onclick = () => downloadCityReport(record);
  $("downloadCityCsv").onclick = () => downloadCsv("gridf_municipal_idf_values.csv", rows.map((row) => ({municipality: record.name, municipality_code: record.code, state: record.state, latitude: record.latitude, longitude: record.longitude, product: cityProductLabel(), disaggregation: cityMethodLabel(), daily_return_depth_mm: params.q24?.[String(state.cityReturnPeriod)] ?? null, mean_valid_area_fraction: params.validAreaFractionMean, coefficient_relative_to_daily: record.disagg?.relative_to_daily?.[String(row.duration_min)] ?? null, coefficient_relative_to_reference_duration: record.disagg?.relative_to_subdaily?.[String(row.duration_min)] ?? null, valid_grid_cells: params.validPixels, touched_grid_cells: params.touchedCells, ...row})));
  refreshIcons();
  if (window.Plotly) {
    const cityTicks = ($("cityIdfCurve")?.clientWidth || 420) < 520 ? [5, 60, 1440] : [5, 30, 60, 360, 1440];
    Plotly.newPlot("cityIdfCurve", RETURN_PERIODS.map((period) => ({x: rows.map((row) => row.duration_min), y: rows.map((row) => row[`rp_${period}_yr_mm_h`]), type: "scatter", mode: "lines+markers", name: `${period} yr`, line: {width: 2.2}, marker: {size: 5}})), {margin: {l: 50, r: 8, t: 8, b: 58}, height: 300, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {type: "log", title: state.lang === "pt" ? "Duração (min)" : "Duration (min)", tickmode: "array", tickvals: cityTicks, ticktext: cityTicks.map(formatDuration), automargin: true}, yaxis: {title: state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", rangemode: "tozero", automargin: true}, legend: {orientation: "h", y: 1.22}, font: {family: "DM Sans", size: 9, color: "#16333f"}}, {displayModeBar: false, responsive: true});
    const coefficientTicks = disaggRows.map((row) => row.duration);
    Plotly.newPlot("cityDisaggChart", [{x: disaggRows.map((row) => row.duration), y: disaggRows.map((row) => row.daily), type: "scatter", mode: "lines+markers", name: t("cityDailyCoefficient"), line: {width: 2.2, color: "#0B81A2"}, marker: {size: 5, color: "#0B81A2"}}, {x: disaggRows.map((row) => row.duration), y: disaggRows.map((row) => row.reference), type: "scatter", mode: "lines+markers", name: t("cityReferenceCoefficient"), line: {width: 2.2, color: "#EA801C"}, marker: {size: 5, color: "#EA801C"}}], {margin: {l: 46, r: 8, t: 8, b: 78}, height: 255, paper_bgcolor: "transparent", plot_bgcolor: "#edf3f0", xaxis: {type: "log", title: state.lang === "pt" ? "Duração" : "Duration", tickmode: "array", tickvals: coefficientTicks, ticktext: coefficientTicks.map(formatDuration), tickangle: -38, automargin: true}, yaxis: {title: state.lang === "pt" ? "Coeficiente" : "Coefficient", rangemode: "tozero", automargin: true}, legend: {orientation: "h", y: 1.24}, font: {family: "DM Sans", size: 8.5, color: "#16333f"}}, {displayModeBar: false, responsive: true});
  }
}

function selectCity(code) { const record = cityRecord(code); if (!record) return; state.citySelected = record; renderCityMap(); renderCityDetail(); }
function renderCities() { if (!state.data.cityCatalog || !state.data.cities) return; renderCityMap(); renderCityDetail(); }

async function loadAtlasMunicipalBoundaries() {
  if (!state.map || state.atlasMunicipalLayer) return;
  try {
    if (!state.data.cityBoundaries) state.data.cityBoundaries = await loadJson("cities-idf-boundaries.geojson");
    state.atlasMunicipalLayer = L.geoJSON(state.data.cityBoundaries, {
      pane: "atlasMunicipalPane",
      interactive: false,
      style: {color: "#264653", weight: .32, opacity: .26, fillOpacity: 0}
    }).addTo(state.map);
    state.states?.bringToFront();
    state.boundary?.bringToFront();
  } catch (error) {
    console.warn("Municipal boundary overlay was not loaded", error);
  }
}

async function loadCityView() {
  if (!state.data.cities || !state.data.cityCatalog) {
    if (!state.cityDataPromise) {
      $("cityStatus").textContent = t("cityLoading");
      state.cityDataPromise = Promise.all([loadJson("cities.geojson"), loadJson("city-catalog.json")]).then(([cities, cityCatalog]) => {
        state.data.cities = cities;
        state.data.cityCatalog = cityCatalog;
        populateSelects();
        initCityMap();
        renderCities();
      }).catch((error) => {
        console.error(error);
        $("cityStatus").textContent = t("rasterError");
        showToast(t("rasterError"));
      });
    }
    await state.cityDataPromise;
  }
  if (state.cityMap) {
    setTimeout(() => { state.cityMap.invalidateSize(); if (state.cityLayer) state.cityMap.fitBounds(state.cityLayer.getBounds(), {padding: [14, 14]}); }, 120);
  }
}

function populateSelects() {
  fillSelect("idfProduct", PRODUCT_OPTIONS, state.product, state.lang);
  fillSelect("idfMethod", METHOD_OPTIONS, state.method, state.lang);
  fillSelect("idfLayer", LAYER_OPTIONS, state.layer, state.lang);
  fillSelect("idfDuration", DURATION_VALUES.map((value) => ({value, label: formatDuration(value)})), state.duration);
  fillSelect("idfReturnPeriod", RETURN_PERIODS.map((value) => ({value, label: `${value} yr`})), state.returnPeriod);
  const disaggRecords = state.data.disaggCatalog?.records || [];
  fillSelect("disaggFamily", DISAGG_FAMILY_OPTIONS, state.disaggFamily, state.lang);
  const availableDurations = disaggRecords.filter((entry) => entry.family === state.disaggFamily).map((entry) => ({value: entry.durationMin, label: state.lang === "pt" ? entry.durationLabelPt : entry.durationLabel}));
  fillSelect("disaggDuration", availableDurations, state.disaggDuration);
  if (!availableDurations.some((item) => Number(item.value) === Number(state.disaggDuration)) && availableDurations.length) state.disaggDuration = Number(availableDurations[0].value);
  if ($("cityProduct")) {
    fillSelect("cityProduct", PRODUCT_OPTIONS, state.cityProduct, state.lang);
    fillSelect("cityMethod", CITY_METHOD_OPTIONS, state.cityMethod, state.lang);
    fillSelect("cityDuration", DURATION_VALUES.map((value) => ({value, label: formatDuration(value)})), state.cityDuration);
    fillSelect("cityReturnPeriod", RETURN_PERIODS.map((value) => ({value, label: `${value} yr`})), state.cityReturnPeriod);
    const states = [...new Map(cityRecords().map((record) => [record.stateCode, record.state])).entries()].sort((left, right) => left[1].localeCompare(right[1], state.lang === "pt" ? "pt-BR" : "en-US"));
    fillSelect("cityState", [{value: "all", label: t("cityStateAll")}, ...states.map(([value, label]) => ({value, label}))], state.cityState);
    $("citySearch").value = state.citySearch || "";
  }
  syncPlotReturnInputs();
}

function setLanguage(language) {
  state.lang = language;
  document.documentElement.lang = language === "pt" ? "pt-BR" : "en";
  document.querySelectorAll("[data-i18n]").forEach((element) => { const key = element.dataset.i18n; if (COPY[language][key]) element.textContent = COPY[language][key]; });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((element) => { element.placeholder = COPY[language][element.dataset.i18nPlaceholder] || element.placeholder; });
  document.querySelectorAll(".language-button").forEach((button) => { const active = button.dataset.language === language; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); });
  populateSelects();
  renderAtlasMeta();
  if (state.selected) renderIdfDetail();
  else if (state.view === "atlas") renderIdfEmpty();
  if (state.view === "disagg") { renderDisaggMap(); renderDisaggDetail(); renderDisaggTable(); }
  if (state.view === "methods") renderMethods();
  if (state.view === "cities") renderCities();
  refreshIcons();
}

function setView(view) {
  state.view = view;
  $("atlasControls")?.classList.remove("open");
  $("cityControls")?.classList.remove("open");
  document.querySelectorAll(".nav-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  ["atlas", "disagg", "cities", "methods"].forEach((name) => { const element = $(`${name}View`); if (element) element.hidden = name !== view; });
  if (view === "disagg") setTimeout(() => { state.disaggMap?.invalidateSize(); if (state.disaggBoundary) state.disaggMap.fitBounds(state.disaggBoundary.getBounds(), {padding: [14, 14]}); }, 120);
  if (view === "cities") loadCityView();
  if (view === "disagg") renderDisaggMap();
  if (view === "methods") renderMethods();
  if (view === "cities") renderCities();
  if (view === "atlas") setTimeout(() => state.map?.invalidateSize(), 80);
}

function bindUi() {
  $("idfProduct").onchange = () => { state.product = $("idfProduct").value; state.selected = null; renderRaster(); renderIdfEmpty(); };
  $("idfMethod").onchange = () => { state.method = $("idfMethod").value; state.selected = null; renderRaster(); renderIdfEmpty(); };
  $("idfDuration").onchange = () => { state.duration = Number($("idfDuration").value); renderRaster(); if (state.selected) renderIdfDetail(); };
  $("idfReturnPeriod").onchange = () => { state.returnPeriod = Number($("idfReturnPeriod").value); renderRaster(); if (state.selected) renderIdfDetail(); };
  $("idfLayer").onchange = () => { state.layer = $("idfLayer").value; renderRaster(); };
  bindPlotReturnInput("idfPlotReturnPeriods", "plotReturnPeriods", () => { if (state.selected) renderIdfDetail(); });
  $("disaggFamily").onchange = () => { state.disaggFamily = $("disaggFamily").value; populateSelects(); renderDisaggMap(); };
  $("disaggDuration").onchange = () => { state.disaggDuration = Number($("disaggDuration").value); renderDisaggSurface(); };
  $("citySearch").oninput = () => { state.citySearch = $("citySearch").value; renderCityMap(); };
  $("cityState").onchange = () => { state.cityState = $("cityState").value; state.citySelected = null; renderCities(); };
  $("cityProduct").onchange = () => { state.cityProduct = $("cityProduct").value; state.citySelected = null; renderCities(); };
  $("cityMethod").onchange = () => { state.cityMethod = $("cityMethod").value; state.citySelected = null; renderCities(); };
  $("cityDuration").onchange = () => { state.cityDuration = Number($("cityDuration").value); renderCities(); };
  $("cityReturnPeriod").onchange = () => { state.cityReturnPeriod = Number($("cityReturnPeriod").value); renderCities(); };
  bindPlotReturnInput("cityPlotReturnPeriods", "cityPlotReturnPeriods", () => { if (state.citySelected) renderCityDetail(); });
  $("downloadCityCatalogButton").onclick = () => downloadCityCatalog();
  document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  document.querySelectorAll(".language-button").forEach((button) => button.addEventListener("click", () => setLanguage(button.dataset.language)));
  $("closeWelcomeButton").onclick = () => { $("welcomeModal").hidden = true; };
  $("helpButton").onclick = () => { $("welcomeModal").hidden = false; setWelcomeStep(0); };
  $("welcomeBack").onclick = () => setWelcomeStep(state.welcomeStep - 1);
  $("welcomeNext").onclick = () => state.welcomeStep === 3 ? ($("welcomeModal").hidden = true) : setWelcomeStep(state.welcomeStep + 1);
  document.querySelectorAll(".welcome-dot").forEach((button) => button.addEventListener("click", () => setWelcomeStep(Number(button.dataset.step))));
  $("mobileControlsButton").onclick = () => {
    const target = state.view === "cities" ? $("cityControls") : $("atlasControls");
    target?.classList.toggle("open");
  };
  $("downloadDisaggButton").onclick = () => downloadCsv("gridf_disaggregation_coefficients.csv", state.data.stations.records);
  $("downloadIdfGridButton").onclick = async () => { const entry = selectedCatalogEntry(); try { const response = await fetch(`${DATA_PATH}${entry.file}`); downloadBlob(`gridf_${entry.product}_${entry.method}_BC_parameters.tif`, await response.blob()); } catch (error) { showToast(t("rasterError")); } };
}

async function createApp() {
  try {
    state.data = {stations: await loadJson("stations.json"), brazil: await loadJson("brazil.geojson"), states: await loadJson("states.geojson"), idfCatalog: await loadJson("idf-catalog.json"), disaggCatalog: await loadJson("disagg-catalog.json"), catalog: await loadJson("catalog.json")};
    state.cityProduct = state.product;
    state.cityMethod = "local-interpolated";
    populateSelects();
    bindUi();
    initAtlasMap();
    initDisaggMap();
    renderIdfEmpty();
    renderMethods();
    renderAtlasMeta();
    renderDisaggDetail();
    setWelcomeStep(0);
    refreshIcons();
    await renderRaster();
    renderDisaggMap();
    loadAtlasMunicipalBoundaries();
  } catch (error) { console.error(error); showToast(error.message); }
}

// Keep the public product name separate from the archived raster directory name.
COPY.en.productText = "The local browser bundle includes bias-corrected parameter stacks for BR-DWGD, IMERG, CHIRPS, and PERSIANN-CDR. The interface retains the three archived disaggregation pathways: local/interpolated, CETESB fixed ratios, and station-derived ratios.";
COPY.pt.productText = "O pacote local inclui pilhas de parâmetros corrigidas por viés para BR-DWGD, IMERG, CHIRPS e PERSIANN-CDR. A interface mantém os três caminhos de desagregação arquivados: local/interpolado, razões fixas CETESB e derivado de estações.";
COPY.en.stationText = "The Disaggregation tab displays the station coefficient export and the interpolated coefficient surfaces used by the IDF curves.";
COPY.pt.stationText = "A aba Desagregação exibe o exportador de coeficientes das estações e as superfícies interpoladas usadas pelas curvas IDF.";
COPY.en.gridNativeNote = "The selected product is displayed and sampled on its own native grid; clicks outside Brazil are rejected.";
COPY.pt.gridNativeNote = "O produto selecionado é exibido e amostrado em sua própria grade nativa; cliques fora do Brasil são rejeitados.";
COPY.en.outsideBrazil = "Please click within Brazil.";
COPY.pt.outsideBrazil = "Clique dentro do Brasil.";
COPY.en.cityLoading = "Loading municipal boundaries…";
COPY.pt.cityLoading = "Carregando limites municipais…";

// Keep the national boundary visible above the raster image overlays.
const baseRenderRaster = renderRaster;
renderRaster = async function (...args) {
  const result = await baseRenderRaster(...args);
  state.atlasMunicipalLayer?.bringToFront();
  state.states?.bringToFront();
  state.boundary?.bringToFront();
  return result;
};

const baseRenderDisaggSurface = renderDisaggSurface;
renderDisaggSurface = async function (...args) {
  const result = await baseRenderDisaggSurface(...args);
  state.disaggBoundary?.bringToFront();
  state.stationLayer?.eachLayer((layer) => layer.bringToFront?.());
  return result;
};

// Use each GeoTIFF's own affine footprint for both drawing and sampling. The
// products are all geographic rasters, but their native origins and extents
// are not identical, so a shared origin would select the wrong cell.
function rasterPixelIndex(raster, lat, lon) {
  if (!raster || !Array.isArray(raster.bounds)) return null;
  const [west, south, east, north] = raster.bounds.map(Number);
  const width = Number(raster.width);
  const height = Number(raster.height);
  if (![west, south, east, north, width, height].every(Number.isFinite)) return null;
  if (east <= west || north <= south || width < 1 || height < 1) return null;
  const xFraction = (Number(lon) - west) / (east - west);
  const yFraction = (north - Number(lat)) / (north - south);
  if (!Number.isFinite(xFraction) || !Number.isFinite(yFraction) || xFraction < 0 || xFraction >= 1 || yFraction < 0 || yFraction >= 1) return null;
  const x = Math.floor(xFraction * width);
  const y = Math.floor(yFraction * height);
  if (x < 0 || y < 0 || x >= width || y >= height) return null;
  return {x, y};
}

function pointOnSegment(lon, lat, first, second) {
  const epsilon = 1e-10;
  const cross = (lat - first[1]) * (second[0] - first[0]) - (lon - first[0]) * (second[1] - first[1]);
  if (Math.abs(cross) > epsilon) return false;
  return lon >= Math.min(first[0], second[0]) - epsilon && lon <= Math.max(first[0], second[0]) + epsilon && lat >= Math.min(first[1], second[1]) - epsilon && lat <= Math.max(first[1], second[1]) + epsilon;
}

function pointInRing(lon, lat, ring) {
  if (!Array.isArray(ring) || ring.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index, index += 1) {
    const currentPoint = ring[index];
    const previousPoint = ring[previous];
    if (pointOnSegment(lon, lat, previousPoint, currentPoint)) return true;
    const intersects = ((currentPoint[1] > lat) !== (previousPoint[1] > lat)) && lon < ((previousPoint[0] - currentPoint[0]) * (lat - currentPoint[1])) / (previousPoint[1] - currentPoint[1]) + currentPoint[0];
    if (intersects) inside = !inside;
  }
  return inside;
}

function pointInPolygonCoordinates(lon, lat, coordinates) {
  if (!Array.isArray(coordinates) || !pointInRing(lon, lat, coordinates[0])) return false;
  return !coordinates.slice(1).some((ring) => pointInRing(lon, lat, ring));
}

function pointInGeometry(lon, lat, value) {
  const geometry = value?.geometry || value;
  if (!geometry) return false;
  if (geometry.type === "FeatureCollection") return (geometry.features || []).some((feature) => pointInGeometry(lon, lat, feature));
  if (geometry.type === "Feature") return pointInGeometry(lon, lat, geometry.geometry);
  if (geometry.type === "Polygon") return pointInPolygonCoordinates(lon, lat, geometry.coordinates);
  if (geometry.type === "MultiPolygon") return (geometry.coordinates || []).some((polygon) => pointInPolygonCoordinates(lon, lat, polygon));
  if (geometry.type === "GeometryCollection") return (geometry.geometries || []).some((item) => pointInGeometry(lon, lat, item));
  return false;
}

function isPointInBrazil(lon, lat) {
  return pointInGeometry(Number(lon), Number(lat), state.data?.brazil);
}

// Replace the original sample routine after the final app definitions have
// loaded, so the click path uses the same native-grid geometry as the overlay.
sampleRaster = function (raster, lat, lon) {
  const index = rasterPixelIndex(raster, lat, lon);
  if (!index) return null;
  return rasterParametersAt(raster, index.x, index.y);
};

const baseHandleAtlasClick = handleAtlasClick;
handleAtlasClick = async function (event) {
  const latitude = event?.latlng?.lat;
  const longitude = event?.latlng?.lng;
  if (!isPointInBrazil(longitude, latitude)) {
    showToast(t("outsideBrazil"));
    return;
  }
  return baseHandleAtlasClick(event);
};

function cityReportRings(value) {
  const geometry = value?.geometry || value;
  if (!geometry) return [];
  if (geometry.type === "FeatureCollection") return geometry.features.flatMap(cityReportRings);
  if (geometry.type === "Feature") return cityReportRings(geometry.geometry);
  if (geometry.type === "Polygon") return geometry.coordinates || [];
  if (geometry.type === "MultiPolygon") return (geometry.coordinates || []).flat();
  return [];
}

function cityReportBounds(value) {
  const points = cityReportRings(value).flat();
  if (!points.length) return [-74, -34, -28, 6];
  let minLon = Infinity;
  let minLat = Infinity;
  let maxLon = -Infinity;
  let maxLat = -Infinity;
  points.forEach((point) => {
    const longitude = Number(point[0]);
    const latitude = Number(point[1]);
    minLon = Math.min(minLon, longitude);
    minLat = Math.min(minLat, latitude);
    maxLon = Math.max(maxLon, longitude);
    maxLat = Math.max(maxLat, latitude);
  });
  return [minLon, minLat, maxLon, maxLat];
}

function cityReportDrawGeometry(context, value, bounds, box, fill, stroke, lineWidth) {
  const rings = cityReportRings(value);
  if (!rings.length) return;
  const [minLon, minLat, maxLon, maxLat] = bounds;
  const [left, top, width, height] = box;
  const project = (point) => [left + ((Number(point[0]) - minLon) / (maxLon - minLon || 1)) * width, top + height - ((Number(point[1]) - minLat) / (maxLat - minLat || 1)) * height];
  context.save();
  context.beginPath();
  rings.forEach((ring) => ring.forEach((point, index) => {
    const [x, y] = project(point);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }));
  if (fill) { context.fillStyle = fill; context.fill("evenodd"); }
  if (stroke) { context.strokeStyle = stroke; context.lineWidth = lineWidth || 1; context.stroke(); }
  context.restore();
}

function cityReportCurveFigure(record, params) {
  const scale = 5;
  const width = 780;
  const height = 480;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#176b78";
  context.fillRect(0, 0, width, 56);
  context.fillStyle = "#ffffff";
  context.font = "700 19px DM Sans, sans-serif";
  context.fillText(t("cityReportFigureCurve"), 32, 28);
  context.font = "400 12px DM Sans, sans-serif";
  context.fillText(record.name + " · " + cityProductLabel() + " · Gumbel", 32, 46);
  const rows = curveRows(params).filter((row) => Number.isFinite(Number(row.duration_min)));
  const durations = rows.map((row) => Number(row.duration_min));
  const values = RETURN_PERIODS.flatMap((period) => rows.map((row) => Number(row["rp_" + period + "_yr_mm_h"]))).filter(Number.isFinite);
  const x0 = 76;
  const y0 = 92;
  const plotWidth = 654;
  const plotHeight = 300;
  const maxValue = Math.max(...values, 1) * 1.08;
  const minDuration = Math.min(...durations, 5);
  const maxDuration = Math.max(...durations, 1440);
  const xPosition = (duration) => x0 + (Math.log(duration) - Math.log(minDuration)) / (Math.log(maxDuration) - Math.log(minDuration) || 1) * plotWidth;
  const yPosition = (value) => y0 + plotHeight - (value / maxValue) * plotHeight;
  context.fillStyle = "#edf3f0";
  context.fillRect(x0, y0, plotWidth, plotHeight);
  context.strokeStyle = "#cbd9d5";
  context.lineWidth = 1;
  context.fillStyle = "#647b82";
  context.font = "400 11px DM Sans, sans-serif";
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const y = y0 + plotHeight - fraction * plotHeight;
    context.beginPath(); context.moveTo(x0, y); context.lineTo(x0 + plotWidth, y); context.stroke();
    context.fillText(Math.round(maxValue * fraction).toLocaleString(state.lang === "pt" ? "pt-BR" : "en-US"), 18, y + 4);
  });
  [5, 10, 30, 60, 360, 1440].forEach((duration) => {
    const x = xPosition(duration);
    context.beginPath(); context.moveTo(x, y0); context.lineTo(x, y0 + plotHeight); context.stroke();
    context.fillText(formatDuration(duration), x - 15, y0 + plotHeight + 20);
  });
  context.strokeStyle = "#16333f";
  context.lineWidth = 1.4;
  context.beginPath(); context.moveTo(x0, y0); context.lineTo(x0, y0 + plotHeight); context.lineTo(x0 + plotWidth, y0 + plotHeight); context.stroke();
  context.save();
  context.translate(15, y0 + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillStyle = "#16333f";
  context.font = "600 12px DM Sans, sans-serif";
  context.fillText(state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", 0, 0);
  context.restore();
  context.fillStyle = "#16333f";
  context.font = "600 12px DM Sans, sans-serif";
  context.fillText(state.lang === "pt" ? "Duração (escala logarítmica)" : "Duration (logarithmic scale)", x0 + 220, y0 + plotHeight + 43);
  const colors = ["#0B81A2", "#E25759", "#59A89C", "#F0C571", "#7E4794", "#9D2C00", "#36B700"];
  RETURN_PERIODS.forEach((period, seriesIndex) => {
    context.strokeStyle = colors[seriesIndex % colors.length];
    context.fillStyle = colors[seriesIndex % colors.length];
    context.lineWidth = 2.2;
    context.beginPath();
    let started = false;
    rows.forEach((row) => {
      const value = Number(row["rp_" + period + "_yr_mm_h"]);
      if (!Number.isFinite(value)) return;
      const x = xPosition(Number(row.duration_min));
      const y = yPosition(value);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    });
    context.stroke();
    rows.forEach((row) => {
      const value = Number(row["rp_" + period + "_yr_mm_h"]);
      if (!Number.isFinite(value)) return;
      context.beginPath(); context.arc(xPosition(Number(row.duration_min)), yPosition(value), 2.6, 0, Math.PI * 2); context.fill();
    });
    const legendX = 82 + (seriesIndex % 5) * 124;
    const legendY = 72 + Math.floor(seriesIndex / 5) * 16;
    context.fillRect(legendX, legendY - 7, 24, 3);
    context.fillStyle = "#16333f";
    context.font = "400 10px DM Sans, sans-serif";
    context.fillText(period + " yr", legendX + 30, legendY);
  });
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

function cityReportCoefficientFigure(record, family) {
  const scale = 5;
  const width = 780;
  const height = 430;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  const rows = cityDisaggRows(record);
  const values = rows.map((row) => Number(family === "daily" ? row.daily : row.reference)).filter(Number.isFinite);
  const title = family === "daily" ? t("cityReportDisaggDailyFigure") : t("cityReportDisaggReferenceFigure");
  const yLabel = state.lang === "pt" ? "Coeficiente" : "Coefficient";
  const xLabel = state.lang === "pt" ? "Duração" : "Duration";
  const lineColor = family === "daily" ? "#0B81A2" : "#EA801C";
  const markerColor = family === "daily" ? "#0B81A2" : "#EA801C";
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#176B78";
  context.fillRect(0, 0, width, 58);
  context.fillStyle = "#ffffff";
  context.font = "700 18px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(title, 32, 28);
  context.font = "400 11px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(record.name + " · " + cityProductLabel() + " · " + (family === "daily" ? (state.lang === "pt" ? "referência diária" : "daily reference") : (state.lang === "pt" ? "razões relativas à duração de referência" : "reference-duration ratios")), 32, 46);
  const x0 = 86;
  const y0 = 88;
  const plotWidth = 638;
  const plotHeight = 270;
  const maxValue = Math.max(...values, 1) * 1.14;
  const xPosition = (index) => x0 + (index / Math.max(rows.length - 1, 1)) * plotWidth;
  const yPosition = (value) => y0 + plotHeight - (value / maxValue) * plotHeight;
  context.fillStyle = "#edf3f0";
  context.fillRect(x0, y0, plotWidth, plotHeight);
  context.strokeStyle = "#D4D4D4";
  context.lineWidth = 1;
  context.fillStyle = "#566B72";
  context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const y = y0 + plotHeight - fraction * plotHeight;
    context.beginPath(); context.moveTo(x0, y); context.lineTo(x0 + plotWidth, y); context.stroke();
    context.fillText((maxValue * fraction).toFixed(2), 42, y + 4);
  });
  context.strokeStyle = "#262626";
  context.lineWidth = 1.6;
  context.beginPath(); context.moveTo(x0, y0); context.lineTo(x0, y0 + plotHeight); context.lineTo(x0 + plotWidth, y0 + plotHeight); context.stroke();
  context.save();
  context.translate(18, y0 + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillStyle = "#262626";
  context.font = "600 12px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(yLabel, 0, 0);
  context.restore();
  context.fillStyle = "#262626";
  context.font = "600 12px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(xLabel, x0 + plotWidth / 2 - 24, height - 18);
  context.strokeStyle = lineColor;
  context.fillStyle = markerColor;
  context.lineWidth = 2.6;
  context.beginPath();
  rows.forEach((row, index) => {
    const value = Number(family === "daily" ? row.daily : row.reference);
    if (!Number.isFinite(value)) return;
    const x = xPosition(index);
    const y = yPosition(value);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  rows.forEach((row, index) => {
    const value = Number(family === "daily" ? row.daily : row.reference);
    if (!Number.isFinite(value)) return;
    const x = xPosition(index);
    const y = yPosition(value);
    context.beginPath(); context.arc(x, y, 3.4, 0, Math.PI * 2); context.fill();
    context.strokeStyle = "#fffdf8"; context.lineWidth = 1.2; context.stroke(); context.strokeStyle = lineColor;
    const label = formatDuration(row.duration).replace(" ", "\n");
    context.fillStyle = "#3C5158";
    context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
    const parts = label.split("\n");
    parts.forEach((part, partIndex) => context.fillText(part, x - context.measureText(part).width / 2, y0 + plotHeight + 18 + partIndex * 12));
  });
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

async function cityReportLogoFigure() {
  const response = await fetch("assets/logos/logo-rain-grid.svg");
  if (!response.ok) throw new Error("Unable to load GRIDF-BR logo");
  const svg = await response.text();
  const url = URL.createObjectURL(new Blob([svg], {type: "image/svg+xml"}));
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = reject;
      element.src = url;
    });
    const canvas = document.createElement("canvas");
    canvas.width = 1920;
    canvas.height = 480;
    const context = canvas.getContext("2d");
    context.fillStyle = "#fffdf8";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 24, 24, 1872, 432);
    return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
  } finally {
    URL.revokeObjectURL(url);
  }
}

function cityReportMapFigure(record) {
  const scale = 5;
  const width = 780;
  const height = 440;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, width, height);
  context.fillStyle = "#176b78";
  context.fillRect(0, 0, width, 56);
  context.fillStyle = "#ffffff";
  context.font = "700 19px DM Sans, sans-serif";
  context.fillText(t("cityReportFigureMap"), 32, 28);
  context.font = "400 12px DM Sans, sans-serif";
  context.fillText(record.name + " · " + record.state + " · " + fmt(record.latitude, 4) + ", " + fmt(record.longitude, 4), 32, 46);
  const nationalBounds = [-74.2, -34.2, -28.4, 5.7];
  const feature = state.data.cities?.features?.find((item) => String(item.properties?.code) === String(record.code));
  cityReportDrawGeometry(context, state.data.brazil, nationalBounds, [36, 86, 310, 290], "#d9ebe6", "#557c7b", 1.2);
  const nationalX = 36 + ((record.longitude - nationalBounds[0]) / (nationalBounds[2] - nationalBounds[0])) * 310;
  const nationalY = 86 + 290 - ((record.latitude - nationalBounds[1]) / (nationalBounds[3] - nationalBounds[1])) * 290;
  context.fillStyle = "#e88a32";
  context.beginPath(); context.arc(nationalX, nationalY, 6, 0, Math.PI * 2); context.fill();
  context.strokeStyle = "#ffffff"; context.lineWidth = 2; context.stroke();
  const localBounds = cityReportBounds(feature || state.data.brazil);
  const lonPad = Math.max((localBounds[2] - localBounds[0]) * .18, .01);
  const latPad = Math.max((localBounds[3] - localBounds[1]) * .18, .01);
  const localBox = [400, 86, 310, 290];
  cityReportDrawGeometry(context, feature, [localBounds[0] - lonPad, localBounds[1] - latPad, localBounds[2] + lonPad, localBounds[3] + latPad], localBox, "#e9ad61", "#16333f", 1.6);
  context.fillStyle = "#16333f";
  context.font = "700 14px DM Sans, sans-serif";
  context.fillText(record.name, 400, 402);
  context.font = "400 11px DM Sans, sans-serif";
  context.fillStyle = "#647b82";
  context.fillText(state.lang === "pt" ? "Localização nacional" : "National location", 36, 399);
  context.fillText(state.lang === "pt" ? "Polígono municipal selecionado" : "Selected municipal polygon", 400, 422);
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

function dataUrlBytes(dataUrl) {
  const binary = atob(dataUrl.split(",")[1]);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function bytesDataUrl(bytes, mimeType) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 0x8000) binary += String.fromCharCode(...bytes.subarray(index, Math.min(index + 0x8000, bytes.length)));
  return "data:" + mimeType + ";base64," + btoa(binary);
}

function pngCrc32(bytes) {
  let crc = 0xffffffff;
  for (const value of bytes) {
    crc ^= value;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngWriteUint32(bytes, offset, value) {
  bytes[offset] = (value >>> 24) & 255;
  bytes[offset + 1] = (value >>> 16) & 255;
  bytes[offset + 2] = (value >>> 8) & 255;
  bytes[offset + 3] = value & 255;
}

function pngDataUrlWithDpi(dataUrl, dpi) {
  const source = dataUrlBytes(dataUrl);
  const ihdrLength = (source[8] << 24) | (source[9] << 16) | (source[10] << 8) | source[11];
  const insertAt = 8 + 12 + ihdrLength;
  const physicalPixelsPerMetre = Math.round(dpi / 0.0254);
  const chunk = new Uint8Array(21);
  pngWriteUint32(chunk, 0, 9);
  chunk.set([112, 72, 89, 115], 4);
  pngWriteUint32(chunk, 8, physicalPixelsPerMetre);
  pngWriteUint32(chunk, 12, physicalPixelsPerMetre);
  chunk[16] = 1;
  pngWriteUint32(chunk, 17, pngCrc32(chunk.subarray(4, 17)));
  const output = new Uint8Array(source.length + chunk.length);
  output.set(source.subarray(0, insertAt), 0);
  output.set(chunk, insertAt);
  output.set(source.subarray(insertAt), insertAt + chunk.length);
  return bytesDataUrl(output, "image/png");
}

function docxImageParagraph(relId, name, widthInches, heightInches, docPrId) {
  const cx = Math.round(widthInches * 914400);
  const cy = Math.round(heightInches * 914400);
  return "<w:p><w:pPr><w:jc w:val=\"center\"/><w:keepNext/></w:pPr><w:r><w:drawing><wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\"><wp:extent cx=\"" + cx + "\" cy=\"" + cy + "\"/><wp:docPr id=\"" + docPrId + "\" name=\"" + name + "\"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect=\"1\"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri=\"http://schemas.openxmlformats.org/drawingml/2006/picture\"><pic:pic><pic:nvPicPr><pic:cNvPr id=\"" + docPrId + "\" name=\"" + name + "\"/><pic:cNvPicPr><a:picLocks noChangeAspect=\"1\"/></pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed=\"" + relId + "\"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"" + cx + "\" cy=\"" + cy + "\"/></a:xfrm><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>";
}

function docxImageRelationships() {
  return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/gridf-logo.png\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-location.png\"/><Relationship Id=\"rId4\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-idf-curves.png\"/><Relationship Id=\"rId5\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-coeff-daily.png\"/><Relationship Id=\"rId7\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-product-comparison.png\"/></Relationships>";
}

function docPageBreak() {
  return "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>";
}

function docHeadingPageBreak(text, style = "Heading1") {
  return `<w:p><w:pPr><w:pStyle w:val="${style}"/><w:pageBreakBefore/></w:pPr><w:r><w:t>${xmlEscape(text)}</w:t></w:r></w:p>`;
}

function docSectionBreak() {
  return "";
}

function cityReportFileName(record) {
  const slug = fileSlug(record.name);
  return "gridf-municipal-idf-" + slug + ".docx";
}

async function downloadCityReport(record) {
  const params = cityParams(record);
  if (!params || !isFiniteNumber(params.K)) { showToast(t("cityNoData")); return; }
  try {
    const logoFigure = await cityReportLogoFigure();
    const locationFigure = cityReportMapFigure(record);
    const curveFigure = cityReportCurveFigure(record, params);
    const dailyCoefficientFigure = cityReportCoefficientFigure(record, "daily");
    const referenceCoefficientFigure = cityReportCoefficientFigure(record, "reference");
    const selectedRows = [5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => {
      const intensity = idfIntensity(params, duration, state.cityReturnPeriod);
      return [formatDuration(duration), fmt(intensity, 2), fmt(intensity == null ? null : intensity * duration / 60, 2)];
    });
    const disaggRows = cityDisaggRows(record);
    const disaggTable = [[t("duration"), t("cityDailyCoefficient"), t("cityReferenceCoefficient")]].concat(disaggRows.map((row) => [formatDuration(row.duration), fmt(row.daily, 4), fmt(row.reference, 4)]));
    const parametersTable = [
      [t("parameters"), state.lang === "pt" ? "Valor" : "Value"],
      ["K", fmt(params.K, 3)], ["a", fmt(params.a, 4)], ["b", fmt(params.b, 3)], ["c", fmt(params.c, 4)],
      [t("cityDailyDepth"), fmt(params.q24?.[String(state.cityReturnPeriod)], 2) + " mm"],
      [t("cityDailyCoefficient"), fmt(cityCoefficient(record), 4)], [t("cityReferenceCoefficient"), fmt(cityReferenceCoefficient(record), 4)], ["R²", fmt(params.R2, 4)],
      [t("cityYears"), fmt(params.Nyears, 0)], [t("cityCoverage"), fmt(Number(params.validAreaFractionMean) * 100, 1) + "%"], [t("citySupport"), `${fmt(params.validPixels, 0)} / ${fmt(params.touchedCells, 0)}`], [t("cityArea"), fmt(record.areaKm2, 2) + " km²"]
    ];
    const designTable = [[t("duration"), t("intensity") + " (" + t("mmHour") + ")", t("depth") + " (" + t("mm") + ")"]].concat(selectedRows);
    const sections = [
      docxImageParagraph("rId2", "GRIDF-BR logo", 2.7, .675, 1),
      docParagraph("GRIDF-BR · Municipal IDFs", "Eyebrow"),
      docParagraph(t("cityReportTitle") + " — " + record.name, "Title"),
      docParagraph(t("cityReportSubtitle"), "Subtitle"),
      docParagraph(t("cityReportGenerated"), "Small"),
      docHeading(t("cityReportLocation")),
      docParagraph(record.name + ", " + record.state + " (" + record.stateCode + ") · " + fmt(record.latitude, 4) + ", " + fmt(record.longitude, 4)),
      docParagraph(cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel · " + t("biasLabel") + " · " + t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + t("duration") + ": " + formatDuration(state.cityDuration)),
      docxImageParagraph("rId3", "Municipal location", 6.3, 3.55, 2),
      docParagraph(t("cityReportFigureMap") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportParameters")),
      docTable(parametersTable),
      docHeading(t("cityReportMethod")),
      docParagraph(t("cityReportTheory")),
      docParagraph(t("cityReportMunicipal")),
      docParagraph(t("cityReportWorkflow")),
      docHeading(t("cityReportFrequency")),
      docParagraph(t("cityReportFrequencyText")),
      docHeading(t("cityReportDisagg")),
      docParagraph(t("cityReportDisaggNote")),
      docTable(disaggTable),
      docHeading(t("cityReportDisaggDailyFigure")),
      docxImageParagraph("rId5", "Daily disaggregation coefficients", 6.3, 3.47, 3),
      docParagraph(t("cityReportDisaggDailyFigure") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportDisaggReferenceFigure")),
      docxImageParagraph("rId6", "Reference-duration disaggregation coefficients", 6.3, 3.47, 4),
      docParagraph(t("cityReportDisaggReferenceFigure") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportFigureCurve")),
      docxImageParagraph("rId4", "IDF curves", 6.3, 3.88, 5),
      docParagraph(t("cityReportFigureCurve") + ". " + t("cityReportFigureNote"), "Caption"),
      docSectionBreak(),
      docHeading(t("cityReportDesignTable")),
      docParagraph(t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel"),
      docTable(designTable),
      docHeading(t("limitsTitle")),
      docParagraph(t("cityReportInterpretation")),
      "<w:sectPr><w:footerReference w:type=\"default\" r:id=\"rId8\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\"/></w:sectPr>"
    ].join("");
    const documentXml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><w:body>" + sections + "</w:body></w:document>";
    const styles = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:style w:type=\"paragraph\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:pPr><w:spacing w:after=\"145\" w:line=\"285\" w:lineRule=\"auto\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\" w:cs=\"Aptos\"/><w:color w:val=\"16333F\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Eyebrow\"><w:name w:val=\"Eyebrow\"/><w:pPr><w:spacing w:before=\"120\" w:after=\"60\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos Display\" w:hAnsi=\"Aptos Display\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"18\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:pPr><w:keepNext/><w:spacing w:before=\"180\" w:after=\"100\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Georgia\" w:hAnsi=\"Georgia\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"36\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Subtitle\"><w:name w:val=\"Subtitle\"/><w:pPr><w:spacing w:after=\"140\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Small\"><w:name w:val=\"Small\"/><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:color w:val=\"647B82\"/><w:sz w:val=\"16\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Caption\"><w:name w:val=\"Caption\"/><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"45\" w:after=\"180\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"16\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"Heading 1\"/><w:pPr><w:keepNext/><w:spacing w:before=\"300\" w:after=\"120\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Georgia\" w:hAnsi=\"Georgia\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"25\"/></w:rPr></w:style></w:styles>";
    const stylesWithDefaults = styles.replace('<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">', '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Avenir Next" w:hAnsi="Avenir Next" w:cs="Avenir Next"/><w:color w:val="16333F"/><w:sz w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="145" w:line="285" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>').replaceAll("DM Sans", "Avenir Next").replaceAll("Fraunces", "Avenir Next");
    const zip = new JSZip();
    zip.file("[Content_Types].xml", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Default Extension=\"png\" ContentType=\"image/png\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/><Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/><Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/></Types>");
    zip.folder("_rels").file(".rels", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>");
    zip.folder("word").file("document.xml", documentXml);
    zip.folder("word").file("styles.xml", stylesWithDefaults);
    zip.folder("word").file("footer1.xml", docxFooterXml());
    zip.folder("word").folder("_rels").file("document.xml.rels", docxImageRelationships());
    zip.folder("word").folder("media").file("gridf-logo.png", dataUrlBytes(logoFigure));
    zip.folder("word").folder("media").file("city-location.png", dataUrlBytes(locationFigure));
    zip.folder("word").folder("media").file("city-idf-curves.png", dataUrlBytes(curveFigure));
    zip.folder("word").folder("media").file("city-coeff-daily.png", dataUrlBytes(dailyCoefficientFigure));
    zip.folder("word").folder("media").file("city-coeff-reference.png", dataUrlBytes(referenceCoefficientFigure));
    downloadBlob(cityReportFileName(record), await zip.generateAsync({type: "blob", compression: "DEFLATE"}));
    showToast(t("available"));
  } catch (error) {
    console.error(error);
    showToast(t("rasterError"));
  }
}

const baseRenderCityDetail = renderCityDetail;
renderCityDetail = function (...args) {
  const result = baseRenderCityDetail(...args);
  if (state.citySelected && $("downloadCityWord")) $("downloadCityWord").onclick = () => downloadCityReport(state.citySelected);
  return result;
};

Object.assign(COPY.en, {
  cityReportFrequencyText: "Municipal IDFs use the Gumbel distribution fitted by the method of moments for all city-scale frequency estimates.",
  cityReportDisaggReferenceFigure: "Sub-daily coefficients grouped by reference duration",
  cityReportProductFigure: "Rainfall-product comparison",
  cityReportProductCaption: "Selected design intensity across the available bias-corrected rainfall products.",
  cityReportMapLegend: "Selected-intensity scale",
  cityReportNoAnnualSeries: "The municipal catalogue stores fitted Gumbel/Sherman summaries and support diagnostics, but not the full annual municipal series. Empirical frequency plots are therefore not reconstructed in this report.",
  cityReportDisaggNote: "Two temporal-ratio families are reported. The daily family expresses each duration relative to the daily maximum. The sub-daily family uses duration-specific reference denominators, so the figure groups the values by denominator instead of mixing all ratios in one line.",
  citySubdailyReference: "Reference"
});
Object.assign(COPY.pt, {
  cityReportFrequencyText: "As IDFs municipais utilizam a distribuição Gumbel, como mencionado anteriormente, ajustada pelo método dos momentos para todas as estimativas de frequência em escala municipal.",
  cityReportDisaggReferenceFigure: "Coeficientes subdiários agrupados por duração de referência",
  cityReportProductFigure: "Comparação entre produtos de chuva",
  cityReportProductCaption: "Intensidade de projeto selecionada nos produtos de chuva corrigidos por viés disponíveis.",
  cityReportMapLegend: "Escala da intensidade selecionada",
  cityReportNoAnnualSeries: "O catálogo municipal armazena os resumos ajustados de Gumbel/Sherman e os diagnósticos de suporte, mas não a série anual municipal completa. Por isso, gráficos empíricos de frequência não são reconstruídos neste relatório.",
  cityReportDisaggNote: "Duas formas de reportar os coeficientes de desagregação são apresentadas. A primeira expressa cada duração em relação ao máximo diário. A segunda utiliza denominadores de referência específicos por duração; por isso, a figura agrupa os valores por denominador em vez de misturar todas as razões em uma única linha.",
  citySubdailyReference: "Referência"
});

function cityReportLocale() {
  return state.lang === "pt" ? "pt-BR" : "en-US";
}

function cityReportDisaggMeta(family, duration) {
  return (state.data.disaggCatalog?.records || []).find((entry) => entry.family === family && Number(entry.durationMin) === Number(duration));
}

function cityReportReferenceGroups(record) {
  const rows = cityDisaggRows(record)
    .map((row) => {
      const entry = cityReportDisaggMeta("relative_to_subdaily", row.duration);
      const reference = state.lang === "pt" ? entry?.referenceLabelPt : entry?.referenceLabel;
      return {...row, reference: reference || ""};
    })
    .filter((row) => row.reference);
  const order = ["30m", "1h", "24h", "1 day"];
  return order
    .map((reference) => ({reference, rows: rows.filter((row) => row.reference === reference)}))
    .filter((group) => group.rows.length);
}

function cityReportProductRows(record) {
  return PRODUCT_OPTIONS.map((product) => {
    const params = record?.idf?.[product.value + "|" + state.cityMethod];
    return {
      product: product[state.lang],
      params,
      intensity: params ? idfIntensity(params, state.cityDuration, state.cityReturnPeriod) : null
    };
  }).filter((row) => Number.isFinite(Number(row.intensity)));
}

function cityReportDrawAxes(context, box, yMax, yLabel, xLabel) {
  const [x0, y0, plotWidth, plotHeight] = box;
  context.fillStyle = "#edf3f0";
  context.fillRect(x0, y0, plotWidth, plotHeight);
  context.strokeStyle = "#d4d4d4";
  context.lineWidth = 1;
  context.fillStyle = "#566b72";
  context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const y = y0 + plotHeight - fraction * plotHeight;
    context.beginPath();
    context.moveTo(x0, y);
    context.lineTo(x0 + plotWidth, y);
    context.stroke();
    context.fillText((yMax * fraction).toFixed(yMax < 2 ? 2 : 0), x0 - 44, y + 4);
  });
  context.strokeStyle = "#262626";
  context.lineWidth = 1.6;
  context.beginPath();
  context.moveTo(x0, y0);
  context.lineTo(x0, y0 + plotHeight);
  context.lineTo(x0 + plotWidth, y0 + plotHeight);
  context.stroke();
  context.save();
  context.translate(x0 - 58, y0 + plotHeight / 2 + 32);
  context.rotate(-Math.PI / 2);
  context.fillStyle = "#262626";
  context.font = "600 12px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(yLabel, 0, 0);
  context.restore();
  context.fillStyle = "#262626";
  context.font = "600 12px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(xLabel, x0 + plotWidth / 2 - context.measureText(xLabel).width / 2, y0 + plotHeight + 46);
}

function cityReportDrawHeader(context, width, title, subtitle) {
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, width, context.canvas.height);
  context.fillStyle = "#176b78";
  context.fillRect(0, 0, width, 58);
  context.fillStyle = "#ffffff";
  context.font = "700 18px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(title, 32, 28);
  context.font = "400 11px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(subtitle, 32, 46);
}

function cityReportCoefficientFigure(record, family) {
  const scale = 5;
  const width = 780;
  const height = family === "daily" ? 430 : 560;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  const rows = cityDisaggRows(record);
  const title = family === "daily" ? t("cityReportDisaggDailyFigure") : t("cityReportDisaggReferenceFigure");
  cityReportDrawHeader(context, width, title, record.name + " · " + cityProductLabel() + " · " + (family === "daily" ? (state.lang === "pt" ? "maximo diario" : "daily maximum") : (state.lang === "pt" ? "denominadores separados" : "separate denominators")));
  const yLabel = state.lang === "pt" ? "Coeficiente" : "Coefficient";
  const xLabel = state.lang === "pt" ? "Duração" : "Duration";
  if (family === "daily") {
    const values = rows.map((row) => Number(row.daily)).filter(Number.isFinite);
    const x0 = 86;
    const y0 = 88;
    const plotWidth = 638;
    const plotHeight = 270;
    const yMax = Math.max(...values, 1) * 1.14;
    const xPosition = (index) => x0 + (index / Math.max(rows.length - 1, 1)) * plotWidth;
    const yPosition = (value) => y0 + plotHeight - (value / yMax) * plotHeight;
    cityReportDrawAxes(context, [x0, y0, plotWidth, plotHeight], yMax, yLabel, xLabel);
    context.strokeStyle = "#0B81A2";
    context.fillStyle = "#0B81A2";
    context.lineWidth = 2.6;
    context.beginPath();
    rows.forEach((row, index) => {
      const value = Number(row.daily);
      if (!Number.isFinite(value)) return;
      const x = xPosition(index);
      const y = yPosition(value);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
    rows.forEach((row, index) => {
      const value = Number(row.daily);
      if (!Number.isFinite(value)) return;
      const x = xPosition(index);
      const y = yPosition(value);
      context.beginPath();
      context.arc(x, y, 3.4, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = "#fffdf8";
      context.lineWidth = 1.2;
      context.stroke();
      const valueLabel = fmt(value, 2);
      context.font = "700 10px Helvetica Neue, Helvetica, Arial, sans-serif";
      const labelWidth = context.measureText(valueLabel).width;
      const labelY = Math.max(y0 + 13, y - 12);
      context.fillStyle = "rgba(255, 253, 248, .86)";
      context.fillRect(x - labelWidth / 2 - 4, labelY - 11, labelWidth + 8, 14);
      context.fillStyle = "#16333f";
      context.fillText(valueLabel, x - labelWidth / 2, labelY);
      context.fillStyle = "#3c5158";
      context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
      const labelParts = formatDuration(row.duration).split(" ");
      labelParts.forEach((part, partIndex) => context.fillText(part, x - context.measureText(part).width / 2, y0 + plotHeight + 18 + partIndex * 12));
      context.fillStyle = "#0B81A2";
      context.strokeStyle = "#0B81A2";
    });
  } else {
    const groups = cityReportReferenceGroups(record);
    const panelBoxes = [[74, 92, 290, 155], [438, 92, 250, 155], [74, 325, 290, 155], [438, 325, 250, 155]];
    groups.forEach((group, groupIndex) => {
      const box = panelBoxes[groupIndex];
      if (!box) return;
      const values = group.rows.map((row) => Number(row.reference)).filter(Number.isFinite);
      const yMax = Math.max(...values, 1) * 1.18;
      const [x0, y0, plotWidth, plotHeight] = box;
      const xPosition = (index) => x0 + (index / Math.max(group.rows.length - 1, 1)) * plotWidth;
      const yPosition = (value) => y0 + plotHeight - (value / yMax) * plotHeight;
      cityReportDrawAxes(context, box, yMax, yLabel, xLabel);
      context.fillStyle = "#176b78";
      context.font = "700 12px Helvetica Neue, Helvetica, Arial, sans-serif";
      const referenceLabel = group.reference === "1 day" && state.lang === "pt" ? "1 dia" : group.reference;
      context.fillText(t("citySubdailyReference") + ": " + referenceLabel, x0, y0 - 12);
      context.strokeStyle = "#EA801C";
      context.fillStyle = "#EA801C";
      context.lineWidth = 2.4;
      context.beginPath();
      group.rows.forEach((row, index) => {
        const value = Number(row.reference);
        if (!Number.isFinite(value)) return;
        const x = xPosition(index);
        const y = yPosition(value);
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke();
      group.rows.forEach((row, index) => {
        const value = Number(row.reference);
        if (!Number.isFinite(value)) return;
        const x = xPosition(index);
        const y = yPosition(value);
        context.beginPath();
        context.arc(x, y, 3.2, 0, Math.PI * 2);
        context.fill();
        context.strokeStyle = "#fffdf8";
        context.lineWidth = 1.2;
        context.stroke();
        context.fillStyle = "#3c5158";
        context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
        const label = formatDuration(row.duration);
        context.fillText(label, x - context.measureText(label).width / 2, y0 + plotHeight + 18);
        context.fillStyle = "#EA801C";
        context.strokeStyle = "#EA801C";
      });
    });
  }
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

function cityReportProductFigure(record) {
  const scale = 5;
  const width = 780;
  const height = 390;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  const rows = cityReportProductRows(record);
  cityReportDrawHeader(context, width, t("cityReportProductFigure"), `${record.name} · ${formatDuration(state.cityDuration)} · ${state.cityReturnPeriod} yr`);
  const values = rows.map((row) => Number(row.intensity)).filter(Number.isFinite);
  const yMax = Math.max(...values, 1) * 1.16;
  const x0 = 76;
  const y0 = 92;
  const plotWidth = 640;
  const plotHeight = 210;
  cityReportDrawAxes(context, [x0, y0, plotWidth, plotHeight], yMax, state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", state.lang === "pt" ? "Produto de chuva" : "Rainfall product");
  const colors = ["#0B81A2", "#E25759", "#59A89C", "#F0C571"];
  const barWidth = Math.min(92, plotWidth / Math.max(rows.length * 1.65, 1));
  rows.forEach((row, index) => {
    const x = x0 + (index + .5) * (plotWidth / rows.length) - barWidth / 2;
    const barHeight = (Number(row.intensity) / yMax) * plotHeight;
    context.fillStyle = colors[index % colors.length];
    context.fillRect(x, y0 + plotHeight - barHeight, barWidth, barHeight);
    context.fillStyle = "#16333f";
    context.font = "700 10px Helvetica Neue, Helvetica, Arial, sans-serif";
    context.fillText(fmt(row.intensity, 1), x + barWidth / 2 - context.measureText(fmt(row.intensity, 1)).width / 2, y0 + plotHeight - barHeight - 8);
    context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
    const label = row.product.replace("-CDR", "");
    context.fillText(label, x + barWidth / 2 - context.measureText(label).width / 2, y0 + plotHeight + 22);
  });
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

function cityReportMapFigure(record) {
  const scale = 5;
  const width = 780;
  const height = 460;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  cityReportDrawHeader(context, width, t("cityReportFigureMap"), record.name + " · " + record.state + " · " + fmt(record.latitude, 4) + ", " + fmt(record.longitude, 4));
  const nationalBounds = [-74.2, -34.2, -28.4, 5.7];
  const feature = state.data.cities?.features?.find((item) => String(item.properties?.code) === String(record.code));
  const records = cityRecords();
  const scaleValues = cityScale(records);
  context.fillStyle = "#edf3f0";
  context.fillRect(32, 82, 330, 305);
  (state.data.cities?.features || []).forEach((item) => {
    const itemRecord = cityRecord(item.properties?.code);
    const value = itemRecord ? cityIntensity(itemRecord) : null;
    const fill = Number.isFinite(Number(value)) ? cityColor(value, scaleValues.min, scaleValues.max) : "#d4d4d4";
    cityReportDrawGeometry(context, item, nationalBounds, [32, 82, 330, 305], fill, "#ffffff", .18);
  });
  cityReportDrawGeometry(context, state.data.brazil, nationalBounds, [32, 82, 330, 305], null, "#16333f", 1.15);
  cityReportDrawGeometry(context, feature, nationalBounds, [32, 82, 330, 305], "rgba(232,138,50,.72)", "#16333f", 1.6);
  const localBounds = cityReportBounds(feature || state.data.brazil);
  const lonPad = Math.max((localBounds[2] - localBounds[0]) * .18, .01);
  const latPad = Math.max((localBounds[3] - localBounds[1]) * .18, .01);
  const localBox = [420, 86, 288, 255];
  context.fillStyle = "#edf3f0";
  context.fillRect(localBox[0], localBox[1], localBox[2], localBox[3]);
  cityReportDrawGeometry(context, feature, [localBounds[0] - lonPad, localBounds[1] - latPad, localBounds[2] + lonPad, localBounds[3] + latPad], localBox, "#e9ad61", "#16333f", 1.6);
  context.fillStyle = "#16333f";
  context.font = "700 14px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(record.name, 420, 374);
  context.font = "400 11px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillStyle = "#647b82";
  context.fillText(state.lang === "pt" ? "Mapa municipal colorido pela intensidade selecionada" : "Municipal map colored by selected intensity", 32, 412);
  context.fillText(state.lang === "pt" ? "Poligono municipal selecionado" : "Selected municipal polygon", 420, 392);
  context.fillStyle = "#16333f";
  context.font = "700 11px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(t("cityReportMapLegend"), 420, 418);
  const gradient = context.createLinearGradient(420, 432, 708, 432);
  [[0, "#143b5a"], [.25, "#176b78"], [.5, "#55b99f"], [.75, "#f0be62"], [1, "#cf5a53"]].forEach(([stop, color]) => gradient.addColorStop(stop, color));
  context.fillStyle = gradient;
  context.fillRect(420, 430, 288, 10);
  context.fillStyle = "#647b82";
  context.font = "400 10px Helvetica Neue, Helvetica, Arial, sans-serif";
  context.fillText(fmt(scaleValues.min, 1) + " " + t("mmHour"), 420, 454);
  const maxLabel = fmt(scaleValues.max, 1) + " " + t("mmHour");
  context.fillText(maxLabel, 708 - context.measureText(maxLabel).width, 454);
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
}

function docxImageRelationships() {
  return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/gridf-logo.png\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-location.png\"/><Relationship Id=\"rId4\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-idf-curves.png\"/><Relationship Id=\"rId5\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-coeff-daily.png\"/><Relationship Id=\"rId7\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-product-comparison.png\"/></Relationships>";
}

async function downloadCityReport(record) {
  const params = cityParams(record);
  if (!params || !isFiniteNumber(params.K)) { showToast(t("cityNoData")); return; }
  try {
    const logoFigure = await cityReportLogoFigure();
    const locationFigure = cityReportMapFigure(record);
    const curveFigure = cityReportCurveFigure(record, params);
    const dailyCoefficientFigure = cityReportCoefficientFigure(record, "daily");
    const productFigure = cityReportProductFigure(record);
    const selectedRows = [5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => {
      const intensity = idfIntensity(params, duration, state.cityReturnPeriod);
      return [formatDuration(duration), fmt(intensity, 2), fmt(intensity == null ? null : intensity * duration / 60, 2)];
    });
    const disaggRows = cityDisaggRows(record);
    const disaggTable = [[t("duration"), t("cityDailyCoefficient"), t("cityReferenceCoefficient"), t("citySubdailyReference")]].concat(disaggRows.map((row) => {
      const entry = cityReportDisaggMeta("relative_to_subdaily", row.duration);
      return [formatDuration(row.duration), fmt(row.daily, 4), fmt(row.reference, 4), state.lang === "pt" ? entry?.referenceLabelPt || "" : entry?.referenceLabel || ""];
    }));
    const productTable = [[t("dataset"), t("intensity") + " (" + t("mmHour") + ")", "K", "a", "b", "c"]].concat(cityReportProductRows(record).map((row) => [row.product, fmt(row.intensity, 2), fmt(row.params.K, 2), fmt(row.params.a, 4), fmt(row.params.b, 3), fmt(row.params.c, 4)]));
    const parametersTable = [
      [t("parameters"), state.lang === "pt" ? "Valor" : "Value"],
      ["K", fmt(params.K, 3)], ["a", fmt(params.a, 4)], ["b", fmt(params.b, 3)], ["c", fmt(params.c, 4)],
      [t("cityDailyDepth"), fmt(params.q24?.[String(state.cityReturnPeriod)], 2) + " mm"],
      [t("cityDailyCoefficient"), fmt(cityCoefficient(record), 4)], [t("cityReferenceCoefficient"), fmt(cityReferenceCoefficient(record), 4)], ["R2", fmt(params.R2, 4)],
      [t("cityYears"), fmt(params.Nyears, 0)], [t("cityCoverage"), fmt(Number(params.validAreaFractionMean) * 100, 1) + "%"], [t("citySupport"), `${fmt(params.validPixels, 0)} / ${fmt(params.touchedCells, 0)}`], [t("cityArea"), fmt(record.areaKm2, 2) + " km2"]
    ];
    const designTable = [[t("duration"), t("intensity") + " (" + t("mmHour") + ")", t("depth") + " (" + t("mm") + ")"]].concat(selectedRows);
    const sections = [
      docxImageParagraph("rId2", "GRIDF-BR logo", 2.7, .675, 1),
      docParagraph("GRIDF-BR · Municipal IDFs", "Eyebrow"),
      docParagraph(t("cityReportTitle") + " — " + record.name, "Title"),
      docParagraph(t("cityReportSubtitle"), "Subtitle"),
      docParagraph(t("cityReportGenerated"), "Small"),
      docHeading(t("cityReportLocation")),
      docParagraph(record.name + ", " + record.state + " (" + record.stateCode + ") · " + fmt(record.latitude, 4) + ", " + fmt(record.longitude, 4)),
      docParagraph(cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel · " + t("biasLabel") + " · " + t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + t("duration") + ": " + formatDuration(state.cityDuration)),
      docxImageParagraph("rId3", "Municipal intensity map", 6.3, 3.72, 2),
      docParagraph(t("cityReportFigureMap") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportParameters")),
      docTable(parametersTable),
      docHeading(t("cityReportMethod")),
      docParagraph(t("cityReportTheory")),
      docParagraph(t("cityReportMunicipal")),
      docParagraph(t("cityReportWorkflow")),
      docHeading(t("cityReportFrequency")),
      docParagraph(t("cityReportFrequencyText")),
      docParagraph(t("cityReportNoAnnualSeries"), "Small"),
      docxImageParagraph("rId4", "IDF curves", 6.3, 3.88, 3),
      docParagraph(t("cityReportFigureCurve") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportProductFigure")),
      docxImageParagraph("rId7", "Rainfall product comparison", 6.3, 3.15, 4),
      docParagraph(t("cityReportProductCaption") + " " + t("cityReportFigureNote"), "Caption"),
      docTable(productTable),
      docSectionBreak(),
      docHeading(t("cityReportDisagg")),
      docParagraph(t("cityReportDisaggNote")),
      docTable(disaggTable),
      docHeading(t("cityReportDisaggDailyFigure")),
      docxImageParagraph("rId5", "Daily disaggregation coefficients", 6.3, 3.47, 5),
      docParagraph(t("cityReportDisaggDailyFigure") + ". " + t("cityReportFigureNote"), "Caption"),
      docHeading(t("cityReportDisaggReferenceFigure")),
      docxImageParagraph("rId6", "Grouped sub-daily disaggregation coefficients", 6.3, 4.52, 6),
      docParagraph(t("cityReportDisaggReferenceFigure") + ". " + t("cityReportFigureNote"), "Caption"),
      docSectionBreak(),
      docHeading(t("cityReportDesignTable")),
      docParagraph(t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel"),
      docTable(designTable),
      docHeading(t("limitsTitle")),
      docParagraph(t("cityReportInterpretation")),
      "<w:sectPr><w:footerReference w:type=\"default\" r:id=\"rId8\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\"/></w:sectPr>"
    ].join("");
    const documentXml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><w:body>" + sections + "</w:body></w:document>";
    const styles = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:style w:type=\"paragraph\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:pPr><w:spacing w:after=\"145\" w:line=\"285\" w:lineRule=\"auto\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\" w:cs=\"Aptos\"/><w:color w:val=\"16333F\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Eyebrow\"><w:name w:val=\"Eyebrow\"/><w:pPr><w:spacing w:before=\"120\" w:after=\"60\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos Display\" w:hAnsi=\"Aptos Display\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"18\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:pPr><w:keepNext/><w:spacing w:before=\"180\" w:after=\"100\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos Display\" w:hAnsi=\"Aptos Display\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"38\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Subtitle\"><w:name w:val=\"Subtitle\"/><w:pPr><w:spacing w:after=\"140\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Small\"><w:name w:val=\"Small\"/><w:pPr><w:spacing w:after=\"90\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:color w:val=\"647B82\"/><w:sz w:val=\"17\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Caption\"><w:name w:val=\"Caption\"/><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"45\" w:after=\"180\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos\" w:hAnsi=\"Aptos\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"16\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"Heading 1\"/><w:pPr><w:keepNext/><w:spacing w:before=\"300\" w:after=\"120\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Aptos Display\" w:hAnsi=\"Aptos Display\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"26\"/></w:rPr></w:style></w:styles>";
    const zip = new JSZip();
    zip.file("[Content_Types].xml", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Default Extension=\"png\" ContentType=\"image/png\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/><Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/><Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/></Types>");
    zip.folder("_rels").file(".rels", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>");
    zip.folder("word").file("document.xml", documentXml);
    zip.folder("word").file("styles.xml", styles);
    zip.folder("word").folder("_rels").file("document.xml.rels", docxImageRelationships());
    zip.folder("word").folder("media").file("gridf-logo.png", dataUrlBytes(logoFigure));
    zip.folder("word").folder("media").file("city-location.png", dataUrlBytes(locationFigure));
    zip.folder("word").folder("media").file("city-idf-curves.png", dataUrlBytes(curveFigure));
    zip.folder("word").folder("media").file("city-coeff-daily.png", dataUrlBytes(dailyCoefficientFigure));
    zip.folder("word").folder("media").file("city-coeff-reference.png", dataUrlBytes(referenceCoefficientFigure));
    zip.folder("word").folder("media").file("city-product-comparison.png", dataUrlBytes(productFigure));
    downloadBlob(cityReportFileName(record), await zip.generateAsync({type: "blob", compression: "DEFLATE"}));
    showToast(t("available"));
  } catch (error) {
    console.error(error);
    showToast(t("rasterError"));
  }
}

const premiumRenderCityDetail = renderCityDetail;
renderCityDetail = function (...args) {
  const result = premiumRenderCityDetail(...args);
  if (state.citySelected && window.Plotly && $("cityDisaggChart")) {
    const record = state.citySelected;
    const rows = cityDisaggRows(record);
    const ticks = ($("cityDisaggChart")?.clientWidth || 320) < 360 ? [5, 60, 1440] : [5, 30, 60, 360, 1440];
    Plotly.purge("cityDisaggChart");
    Plotly.newPlot("cityDisaggChart", [{
      x: rows.map((row) => row.duration),
      y: rows.map((row) => row.daily),
      type: "scatter",
      mode: "lines+markers",
      name: t("cityDailyCoefficient"),
      line: {width: 2.4, color: "#0B81A2"},
      marker: {size: 5, color: "#0B81A2"}
    }], {
      margin: {l: 48, r: 8, t: 8, b: 64},
      height: 235,
      paper_bgcolor: "transparent",
      plot_bgcolor: "#edf3f0",
      xaxis: {title: state.lang === "pt" ? "Duração" : "Duration", tickmode: "array", tickvals: ticks, ticktext: ticks.map(formatDuration), tickangle: 0, automargin: true},
      yaxis: {title: state.lang === "pt" ? "Coeficiente" : "Coefficient", rangemode: "tozero", automargin: true},
      font: {family: "DM Sans", size: 9, color: "#16333f"},
      showlegend: false
    }, {displayModeBar: false, responsive: true});
  }
  if (state.citySelected && $("downloadCityWord")) $("downloadCityWord").onclick = () => downloadCityReport(state.citySelected);
  return result;
};

var GRIDF_PLOT_MIN_DURATION = 15;
var GRIDF_DEFAULT_CITY_CODE = "BR3550308";

Object.assign(COPY.en, {
  chartCsv: "Chart CSV",
  cityReportTheory: "The IDF relation expresses rainfall intensity as a function of storm duration and return period. In GRIDF-BR, this relation is represented with a four-parameter Sherman equation fitted to the selected municipal design intensities.",
  tableCaptionParameters: "Table 1. Municipal parameters.",
  tableCaptionProducts: "Table 2. Rainfall-product comparison for the selected duration and return period.",
  tableCaptionDisagg: "Table 3. Municipal temporal-distribution coefficients.",
  tableCaptionDesign: "Table 4. Design intensities for the selected return period.",
  cityReportConfiguration: "Selected map configuration",
  cityReportEquationHeading: "IDF equation",
  cityReportProductSubheading: "Rainfall-product comparison",
  cityReportDisaggSubheading: "Temporal-distribution coefficients",
  cityReportDesignSubheading: "Design-intensity table"
});
Object.assign(COPY.pt, {
  chartCsv: "CSV do gráfico",
  cityReportTheory: "A relação IDF expressa a intensidade da chuva em função da duração da tempestade e do período de retorno. Na toolbox do GRIDF-BR, essa relação é representada por uma equação Sherman de quatro parâmetros ajustada às intensidades municipais de projeto selecionadas.",
  tableCaptionParameters: "Tabela 1. Parâmetros municipais.",
  tableCaptionProducts: "Tabela 2. Comparação entre produtos de chuva para a duração e o período de retorno selecionados.",
  tableCaptionDisagg: "Tabela 3. Coeficientes municipais de distribuição temporal.",
  tableCaptionDesign: "Tabela 4. Intensidades de projeto para o período de retorno selecionado.",
  cityReportConfiguration: "Configuração do mapa exibido",
  cityReportEquationHeading: "Equação IDF",
  cityReportProductSubheading: "Comparação entre produtos de chuva",
  cityReportDisaggSubheading: "Coeficientes de distribuição temporal",
  cityReportDesignSubheading: "Tabela de intensidades de projeto"
});
COPY.en.cityReportFigureNote = "";
COPY.pt.cityReportFigureNote = "";

function plotDurationValues() {
  return DURATION_VALUES.filter((duration) => duration >= GRIDF_PLOT_MIN_DURATION);
}

function plotCurveRows(params) {
  return curveRows(params).filter((row) => Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION);
}

function fileSlug(value) {
  return String(value || "location").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "location";
}

function recordSlug(record) {
  return fileSlug(record?.name || record?.code || "municipio");
}

function selectedPointSlug(selected) {
  return fileSlug("lat-" + fmt(selected?.lat, 4) + "-lon-" + fmt(selected?.lon, 4));
}

function chartButton(id) {
  return `<button class="action-button secondary chart-csv-button" id="${id}" type="button"><i data-lucide="download" aria-hidden="true"></i>${t("chartCsv")}</button>`;
}

function cityIdfCsvRows(record) {
  const params = cityParams(record);
  return curveRows(params).map((row) => ({
    municipality: record.name,
    municipality_code: record.code,
    state: record.state,
    product: cityProductLabel(),
    disaggregation: cityMethodLabel(),
    frequency_model: "Gumbel",
    plotted_in_curve: Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION ? 1 : 0,
    ...row
  }));
}

function cityDisaggCsvRows(record) {
  return cityDisaggRows(record).map((row) => {
    const meta = cityReportDisaggMeta("relative_to_subdaily", row.duration);
    return {
      municipality: record.name,
      municipality_code: record.code,
      state: record.state,
      duration_min: row.duration,
      duration_label: formatDuration(row.duration),
      coefficient_relative_to_daily_maximum: row.daily,
      coefficient_relative_to_reference_duration: row.reference,
      reference_duration: state.lang === "pt" ? meta?.referenceLabelPt || "" : meta?.referenceLabel || ""
    };
  });
}

function cityProductCsvRows(record) {
  return cityReportProductRows(record).map((row) => ({
    municipality: record.name,
    municipality_code: record.code,
    state: record.state,
    product: row.product,
    duration_min: state.cityDuration,
    duration_label: formatDuration(state.cityDuration),
    return_period_years: state.cityReturnPeriod,
    frequency_model: "Gumbel",
    intensity_mm_h: row.intensity,
    K: row.params.K,
    a: row.params.a,
    b: row.params.b,
    c: row.params.c,
    R2: row.params.R2,
    Nyears: row.params.Nyears
  }));
}

function formatPlotReturnPeriod(period) {
  const value = Number(period);
  if (!Number.isFinite(value)) return String(period);
  return Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000).replace(/\.?0+$/, "");
}

function plotReturnPeriodLabel(period) {
  return `${formatPlotReturnPeriod(period)} yr`;
}

function formatPlotReturnPeriodInput(periods) {
  return (periods?.length ? periods : DEFAULT_PLOT_RETURN_PERIODS).map(formatPlotReturnPeriod).join(", ");
}

function parsePlotReturnPeriods(text) {
  const periods = String(text || "")
    .split(/[,\s;]+/)
    .map((item) => Number(item.trim()))
    .filter((value) => Number.isFinite(value) && value > 0)
    .map((value) => Math.round(value * 1000) / 1000);
  return [...new Set(periods)].sort((a, b) => a - b).slice(0, 10);
}

function styleForReturnPeriod(period, index = 0) {
  const fixed = IDF_SERIES_STYLES[Number(period)];
  if (fixed) return fixed;
  return {
    color: IDF_STYLE_PALETTE[index % IDF_STYLE_PALETTE.length],
    marker: IDF_MARKERS[index % IDF_MARKERS.length]
  };
}

function activePlotReturnPeriods(scope = "atlas") {
  const periods = scope === "city" ? state.cityPlotReturnPeriods : state.plotReturnPeriods;
  return periods?.length ? periods : DEFAULT_PLOT_RETURN_PERIODS;
}

function curveRowsForPeriods(params, periods) {
  return DURATION_VALUES.map((duration) => {
    const row = {duration: formatDuration(duration), duration_min: duration};
    periods.forEach((period) => {
      const key = `rp_${formatPlotReturnPeriod(period).replace(".", "_")}_yr`;
      const intensity = idfIntensity(params, duration, period);
      row[`${key}_mm_h`] = intensity;
      row[`${key}_mm`] = intensity == null ? null : intensity * duration / 60;
    });
    return row;
  });
}

function bindPlotReturnInput(id, stateKey, renderCallback) {
  const input = $(id);
  if (!input) return;
  input.value = formatPlotReturnPeriodInput(state[stateKey]);
  const apply = () => {
    const periods = parsePlotReturnPeriods(input.value);
    if (!periods.length) {
      input.classList.add("input-invalid");
      showToast(t("plotReturnPeriodsInvalid"));
      return;
    }
    input.classList.remove("input-invalid");
    state[stateKey] = periods;
    input.value = formatPlotReturnPeriodInput(periods);
    renderCallback();
  };
  input.addEventListener("change", apply);
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    input.blur();
    apply();
  });
}

function syncPlotReturnInputs() {
  const atlasInput = $("idfPlotReturnPeriods");
  if (atlasInput && document.activeElement !== atlasInput) atlasInput.value = formatPlotReturnPeriodInput(activePlotReturnPeriods("atlas"));
  const cityInput = $("cityPlotReturnPeriods");
  if (cityInput && document.activeElement !== cityInput) cityInput.value = formatPlotReturnPeriodInput(activePlotReturnPeriods("city"));
}

function plotlyIdfLayout(height = 300, compact = false) {
  const ticks = compact ? [15, 60, 1440] : [15, 30, 60, 360, 1440];
  return {
    margin: {l: 50, r: 8, t: 8, b: 58},
    height,
    paper_bgcolor: "transparent",
    plot_bgcolor: "#edf3f0",
    xaxis: {type: "log", title: state.lang === "pt" ? "Duração (min)" : "Duration (min)", tickmode: "array", tickvals: ticks, ticktext: ticks.map(formatDuration), automargin: true},
    yaxis: {title: state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", rangemode: "tozero", automargin: true},
    legend: {orientation: "h", y: 1.22},
    font: {family: "DM Sans", size: 9, color: "#16333f"}
  };
}

const gridfBaseRenderIdfDetail = renderIdfDetail;
renderIdfDetail = function () {
  gridfBaseRenderIdfDetail();
  const selected = state.selected;
  if (!selected || !window.Plotly || !$("idfCurve")) return;
  const plottedPeriods = activePlotReturnPeriods("atlas");
  const allRows = curveRowsForPeriods(selected.params, plottedPeriods);
  const plotRows = allRows.filter((row) => Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION);
  Plotly.purge("idfCurve");
  Plotly.newPlot("idfCurve", plottedPeriods.map((period, seriesIndex) => {
    const style = styleForReturnPeriod(period, seriesIndex);
    const key = `rp_${formatPlotReturnPeriod(period).replace(".", "_")}_yr_mm_h`;
    return {
      x: plotRows.map((row) => row.duration_min),
      y: plotRows.map((row) => row[key]),
      type: "scatter",
      mode: "lines+markers",
      name: plotReturnPeriodLabel(period),
      line: {width: 2.2, color: style.color},
      marker: {size: 6, color: style.color, symbol: style.marker}
    };
  }), plotlyIdfLayout(300, ($("idfCurve")?.clientWidth || 420) < 520), {displayModeBar: false, responsive: true});
  if (!$("downloadSelectedChartCsv")) {
    $("idfCurve").insertAdjacentHTML("afterend", `<div class="chart-actions">${chartButton("downloadSelectedChartCsv")}</div>`);
  }
  $("downloadSelectedChartCsv").onclick = () => downloadCsv("gridf-idf-" + selectedPointSlug(selected) + "-curva-idf.csv", allRows.map((row) => ({
    latitude: selected.lat,
    longitude: selected.lon,
    product: selectedProductLabel(),
    disaggregation: selectedMethodLabel(),
    plotted_in_curve: Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION ? 1 : 0,
    ...row
  })));
  refreshIcons();
};

const gridfBaseRenderCityDetail = renderCityDetail;
renderCityDetail = function (...args) {
  const result = gridfBaseRenderCityDetail(...args);
  const record = state.citySelected;
  if (!record) return result;
  const params = cityParams(record);
  if (!params) return result;
  const plottedPeriods = activePlotReturnPeriods("city");
  const allRows = curveRowsForPeriods(params, plottedPeriods);
  const plotRows = allRows.filter((row) => Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION);
  if (window.Plotly && $("cityIdfCurve")) {
    Plotly.purge("cityIdfCurve");
    Plotly.newPlot("cityIdfCurve", plottedPeriods.map((period, seriesIndex) => {
      const style = styleForReturnPeriod(period, seriesIndex);
      const key = `rp_${formatPlotReturnPeriod(period).replace(".", "_")}_yr_mm_h`;
      return {
        x: plotRows.map((row) => row.duration_min),
        y: plotRows.map((row) => row[key]),
        type: "scatter",
        mode: "lines+markers",
        name: plotReturnPeriodLabel(period),
        line: {width: 2.2, color: style.color},
        marker: {size: 6, color: style.color, symbol: style.marker}
      };
    }), plotlyIdfLayout(300, ($("cityIdfCurve")?.clientWidth || 420) < 520), {displayModeBar: false, responsive: true});
    if (!$("downloadCityIdfChartCsv")) $("cityIdfCurve").insertAdjacentHTML("afterend", `<div class="chart-actions">${chartButton("downloadCityIdfChartCsv")}</div>`);
    $("downloadCityIdfChartCsv").onclick = () => downloadCsv("gridf-municipal-" + recordSlug(record) + "-curva-idf.csv", allRows.map((row) => ({
      municipality: record.name,
      municipality_code: record.code,
      state: record.state,
      product: cityProductLabel(),
      disaggregation: cityMethodLabel(),
      frequency_model: "Gumbel",
      plotted_in_curve: Number(row.duration_min) >= GRIDF_PLOT_MIN_DURATION ? 1 : 0,
      ...row
    })));
  }
  if ($("cityDisaggChart") && !$("downloadCityDisaggChartCsv")) {
    $("cityDisaggChart").insertAdjacentHTML("afterend", `<div class="chart-actions">${chartButton("downloadCityDisaggChartCsv")}</div>`);
  }
  if ($("downloadCityDisaggChartCsv")) $("downloadCityDisaggChartCsv").onclick = () => downloadCsv("gridf-municipal-" + recordSlug(record) + "-coeficientes-desagregacao.csv", cityDisaggCsvRows(record));
  if ($("downloadCityCsv")) $("downloadCityCsv").onclick = () => downloadCsv("gridf-municipal-" + recordSlug(record) + "-valores-idf.csv", cityIdfCsvRows(record));
  if ($("downloadCityWord")) $("downloadCityWord").onclick = () => downloadCityReport(record);
  refreshIcons();
  return result;
};

function selectDefaultCity() {
  if (!state.citySelected && state.data.cityCatalog && cityRecord(GRIDF_DEFAULT_CITY_CODE)) {
    state.cityState = "all";
    state.citySearch = "";
    populateSelects();
    selectCity(GRIDF_DEFAULT_CITY_CODE);
  }
}

const gridfBaseLoadCityView = loadCityView;
loadCityView = async function (...args) {
  const result = await gridfBaseLoadCityView(...args);
  selectDefaultCity();
  return result;
};

const gridfBaseRenderCities = renderCities;
renderCities = function (...args) {
  const result = gridfBaseRenderCities(...args);
  if (state.view === "cities") selectDefaultCity();
  return result;
};

function docFontPr(font = "Avenir Next") {
  return `<w:rFonts w:ascii="${font}" w:hAnsi="${font}" w:cs="${font}"/>`;
}

function docRun(text, props = "", font = "Avenir Next") {
  return `<w:r><w:rPr>${docFontPr(font)}${props}</w:rPr><w:t xml:space="preserve">${xmlEscape(text)}</w:t></w:r>`;
}

function docMathRun(text) {
  return docRun(text, '<w:i/>');
}

docParagraph = function (text, style = "Normal") {
  const configs = {
    Title: {p: '<w:keepNext/><w:spacing w:before="180" w:after="140"/>', r: '<w:b/><w:color w:val="176B78"/><w:sz w:val="34"/>'},
    Subtitle: {p: '<w:spacing w:after="180"/>', r: '<w:i/><w:color w:val="647B82"/><w:sz w:val="21"/>'},
    Eyebrow: {p: '<w:spacing w:before="120" w:after="60"/>', r: '<w:b/><w:color w:val="176B78"/><w:sz w:val="18"/>'},
    Caption: {p: '<w:jc w:val="center"/><w:spacing w:before="60" w:after="190"/>', r: '<w:b/><w:color w:val="16333F"/><w:sz w:val="18"/>'},
    TableCaption: {p: '<w:jc w:val="center"/><w:spacing w:before="170" w:after="80"/>', r: '<w:b/><w:color w:val="176B78"/><w:sz w:val="18"/>'},
    Small: {p: '<w:spacing w:after="100"/>', r: '<w:color w:val="647B82"/><w:sz w:val="17"/>'},
    Normal: {p: '<w:spacing w:after="145" w:line="285" w:lineRule="auto"/>', r: '<w:color w:val="16333F"/><w:sz w:val="21"/>'}
  };
  const config = configs[style] || configs.Normal;
  return `<w:p><w:pPr><w:pStyle w:val="${style}"/>${config.p}</w:pPr>${docRun(text, config.r)}</w:p>`;
};

docHeading = function (text, style = "Heading1") {
  const isSubheading = style === "Heading2";
  const p = isSubheading ? '<w:keepNext/><w:spacing w:before="210" w:after="95"/>' : '<w:keepNext/><w:spacing w:before="310" w:after="120"/>';
  const r = isSubheading ? '<w:i/><w:color w:val="176B78"/><w:sz w:val="22"/>' : '<w:b/><w:color w:val="176B78"/><w:sz w:val="27"/>';
  return `<w:p><w:pPr><w:pStyle w:val="${style}"/>${p}</w:pPr>${docRun(text, r)}</w:p>`;
}

function docMixedParagraph(parts, style = "Normal") {
  return `<w:p><w:pPr><w:pStyle w:val="${style}"/><w:spacing w:after="145" w:line="285" w:lineRule="auto"/></w:pPr>${parts.map((part) => part.math ? docMathRun(part.text) : docRun(part.text, part.bold ? "<w:b/>" : "")).join("")}</w:p>`;
}

function docEquationParagraph() {
  return '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="120"/></w:pPr><m:oMathPara><m:oMath><m:r><m:rPr><m:nor/></m:rPr><m:t>I</m:t></m:r><m:r><m:t> = </m:t></m:r><m:f><m:fPr><m:type m:val="bar"/></m:fPr><m:num><m:r><m:t>K · </m:t></m:r><m:sSup><m:e><m:r><m:t>RP</m:t></m:r></m:e><m:sup><m:r><m:t>a</m:t></m:r></m:sup></m:sSup></m:num><m:den><m:sSup><m:e><m:r><m:t>(b + t)</m:t></m:r></m:e><m:sup><m:r><m:t>c</m:t></m:r></m:sup></m:sSup></m:den></m:f></m:oMath></m:oMathPara></w:p>';
}

function docTableCaption(number, text) {
  return docParagraph((state.lang === "pt" ? "Tabela " : "Table ") + number + ". " + text, "TableCaption");
}

function docFigureCaption(number, text) {
  return docParagraph((state.lang === "pt" ? "Figura " : "Figure ") + number + ". " + text, "Caption");
}

function cityReportMapCaption(record) {
  const product = cityProductLabel();
  const duration = formatDuration(state.cityDuration);
  const returnPeriod = state.cityReturnPeriod;
  if (state.lang === "pt") {
    return `Distribuição municipal da intensidade IDF do produto ${product}, corrigido por viés, para duração de ${duration} e período de retorno de ${returnPeriod} anos, com a localização de ${record.name}, ${record.state}. No painel esquerdo, cada município é colorido pela intensidade calculada após o ajuste Gumbel e o reajuste da relação Sherman; o município selecionado é destacado em laranja. O painel direito amplia o polígono de ${record.name}. A escala cromática representa intensidade em mm/h entre os percentis 5 e 95 dos municípios com estimativas disponíveis.`;
  }
  return `Municipal distribution of bias-corrected ${product} IDF intensity for a ${duration} duration and ${returnPeriod}-year return period, with the location of ${record.name}, ${record.state}. In the left panel, each municipality is coloured by the intensity calculated after the Gumbel fit and Sherman refit; the selected municipality is highlighted in orange. The right panel enlarges the ${record.name} polygon. The colour scale represents intensity in mm/h between the 5th and 95th percentiles of municipalities with available estimates.`;
}

function cityReportCurveCaption(record) {
  const periods = activePlotReturnPeriods("city").map(formatPlotReturnPeriod).join(", ");
  if (state.lang === "pt") {
    return `Curvas IDF municipais de ${record.name}, ${record.state}, obtidas com a distribuição Gumbel ajustada pelo método dos momentos e a relação Sherman de quatro parâmetros reajustada. Cada cor e símbolo representa um dos períodos de retorno selecionados (${periods} anos); os pontos marcam as durações avaliadas de 15 min a 24 h. O eixo de duração está em escala logarítmica e a intensidade é expressa em mm/h.`;
  }
  return `Municipal IDF curves for ${record.name}, ${record.state}, obtained from the Gumbel distribution fitted by the method of moments and the refitted four-parameter Sherman relation. Each colour and marker represents one selected return period (${periods} years); points mark the evaluated durations from 15 min to 24 h. The duration axis is logarithmic and intensity is expressed in mm/h.`;
}

function cityReportProductCaption(record) {
  const duration = formatDuration(state.cityDuration);
  const returnPeriod = state.cityReturnPeriod;
  if (state.lang === "pt") {
    return `Comparação das intensidades IDF municipais de ${record.name}, ${record.state}, entre os produtos de chuva corrigidos por viés disponíveis, para duração de ${duration} e período de retorno de ${returnPeriod} anos. A altura de cada barra representa a intensidade estimada em mm/h após o ajuste Gumbel e o reajuste da relação Sherman de cada produto; a figura permite inspecionar diferenças entre produtos, sem estabelecer uma classificação universal de desempenho.`;
  }
  return `Comparison of municipal IDF intensities for ${record.name}, ${record.state}, across the available bias-corrected rainfall products, for a ${duration} duration and ${returnPeriod}-year return period. Each bar height is the intensity in mm/h estimated after the Gumbel fit and Sherman refit for that product; the figure supports inspection of product differences without establishing a universal performance ranking.`;
}

function cityReportDailyCoefficientCaption(record) {
  if (state.lang === "pt") {
    return `Coeficientes de desagregação de ${record.name}, ${record.state}, relativos ao máximo diário. Cada ponto é a razão entre a lâmina acumulada na duração indicada e a lâmina diária, obtida das superfícies locais/interpoladas e ponderada pela área do município. Os rótulos mostram o valor do coeficiente; essas razões convertem as lâminas diárias de retorno para as durações subdiárias usadas no reajuste da relação IDF.`;
  }
  return `Disaggregation coefficients for ${record.name}, ${record.state}, relative to the daily maximum. Each point is the ratio between the accumulated depth at the indicated duration and the daily depth, obtained from the local/interpolated surfaces and area-weighted for the municipality. Labels show the coefficient value; these ratios convert daily return depths to the sub-daily durations used to refit the IDF relation.`;
}

function cityReportParametersCaption(record) {
  if (state.lang === "pt") return `Parâmetros e diagnósticos do ajuste municipal de ${record.name}, ${record.state}, para ${cityProductLabel()}. K, a, b e c definem a relação Sherman reajustada; os demais campos descrevem a lâmina diária de retorno, os coeficientes temporais ponderados por área, o ajuste e o suporte espacial utilizado.`;
  return `Parameters and diagnostics of the municipal fit for ${record.name}, ${record.state}, for ${cityProductLabel()}. K, a, b, and c define the refitted Sherman relation; the remaining fields describe the daily return depth, area-weighted temporal coefficients, fit, and spatial support used.`;
}

function cityReportProductTableCaption(record) {
  if (state.lang === "pt") return `Intensidades municipais estimadas para ${record.name}, ${record.state}, por produto de chuva corrigido por viés, para duração de ${formatDuration(state.cityDuration)} e período de retorno de ${state.cityReturnPeriod} anos. Os parâmetros apresentados são os obtidos após o ajuste Gumbel e o reajuste da relação Sherman de cada produto.`;
  return `Municipal intensities estimated for ${record.name}, ${record.state}, by bias-corrected rainfall product, for a ${formatDuration(state.cityDuration)} duration and ${state.cityReturnPeriod}-year return period. The listed parameters were obtained after the Gumbel fit and Sherman refit for each product.`;
}

function cityReportDisaggregationTableCaption(record) {
  if (state.lang === "pt") return `Coeficientes temporais de ${record.name}, ${record.state}, ponderados pela área do município. A coluna relativa ao máximo diário usa a lâmina diária como denominador; a coluna relativa à duração de referência usa o denominador específico indicado na última coluna.`;
  return `Area-weighted temporal coefficients for ${record.name}, ${record.state}. The coefficient relative to the daily maximum uses daily depth as its denominator; the coefficient relative to a reference duration uses the duration-specific denominator shown in the last column.`;
}

function cityReportDesignTableCaption(record) {
  if (state.lang === "pt") return `Intensidades e lâminas de projeto para ${record.name}, ${record.state}, estimadas com ${cityProductLabel()} corrigido por viés, distribuição Gumbel e relação Sherman reajustada, para período de retorno de ${state.cityReturnPeriod} anos. A tabela inclui as durações curtas para consulta numérica.`;
  return `Design intensities and depths for ${record.name}, ${record.state}, estimated with bias-corrected ${cityProductLabel()}, the Gumbel distribution, and the refitted Sherman relation, for a ${state.cityReturnPeriod}-year return period. The table includes short durations for numerical reference.`;
}

docTable = function (rows) {
  return `<w:tbl><w:tblPr><w:jc w:val="center"/><w:tblW w:w="0" w:type="auto"/><w:tblBorders><w:top w:val="single" w:sz="5" w:color="B7C8C3"/><w:left w:val="single" w:sz="5" w:color="B7C8C3"/><w:bottom w:val="single" w:sz="5" w:color="B7C8C3"/><w:right w:val="single" w:sz="5" w:color="B7C8C3"/><w:insideH w:val="single" w:sz="4" w:color="D6DFDC"/><w:insideV w:val="single" w:sz="4" w:color="D6DFDC"/></w:tblBorders><w:tblCellMar><w:top w:w="70" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>${rows.map((row, i) => `<w:tr>${row.map((cell) => `<w:tc><w:tcPr>${i === 0 ? '<w:shd w:fill="176B78"/>' : ""}</w:tcPr><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr>${docFontPr()}${i === 0 ? '<w:b/><w:color w:val="FFFFFF"/>' : ""}</w:rPr><w:t>${xmlEscape(cell)}</w:t></w:r></w:p></w:tc>`).join("")}</w:tr>`).join("")}</w:tbl>`;
};

cityReportDrawHeader = function (context, width, title, subtitle) {
  context.fillStyle = "#fffdf8";
  context.fillRect(0, 0, width, context.canvas.height);
  context.fillStyle = "#176b78";
  context.fillRect(0, 0, width, 58);
  context.fillStyle = "#ffffff";
  context.font = "700 18px DM Sans, Arial, sans-serif";
  context.fillText(title, 32, 28);
  context.font = "400 11px DM Sans, Arial, sans-serif";
  context.fillText(subtitle, 32, 46);
};

function drawIdfCanvasMarker(context, x, y, marker, color, size = 3) {
  context.save();
  context.fillStyle = color;
  context.strokeStyle = color;
  context.lineWidth = 1.6;
  context.beginPath();
  if (marker === "square") context.rect(x - size, y - size, size * 2, size * 2);
  else if (marker === "diamond") { context.moveTo(x, y - size - 1); context.lineTo(x + size + 1, y); context.lineTo(x, y + size + 1); context.lineTo(x - size - 1, y); context.closePath(); }
  else if (marker === "triangle-up") { context.moveTo(x, y - size - 1); context.lineTo(x + size + 1, y + size); context.lineTo(x - size - 1, y + size); context.closePath(); }
  else if (marker === "cross") { context.moveTo(x - size, y); context.lineTo(x + size, y); context.moveTo(x, y - size); context.lineTo(x, y + size); context.stroke(); context.restore(); return; }
  else if (marker === "x") { context.moveTo(x - size, y - size); context.lineTo(x + size, y + size); context.moveTo(x + size, y - size); context.lineTo(x - size, y + size); context.stroke(); context.restore(); return; }
  else if (marker === "star") {
    for (let i = 0; i < 10; i += 1) {
      const angle = -Math.PI / 2 + i * Math.PI / 5;
      const radius = i % 2 === 0 ? size + 1.7 : size * .55;
      const px = x + Math.cos(angle) * radius;
      const py = y + Math.sin(angle) * radius;
      if (i === 0) context.moveTo(px, py); else context.lineTo(px, py);
    }
    context.closePath();
  } else {
    context.arc(x, y, size, 0, Math.PI * 2);
  }
  context.fill();
  context.restore();
}

cityReportCurveFigure = function (record, params) {
  const scale = 5;
  const width = 780;
  const height = 480;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  const reportPeriodLabel = activePlotReturnPeriods("city").map(formatPlotReturnPeriod).join(", ");
  cityReportDrawHeader(context, width, t("cityReportFigureCurve"), record.name + " · " + cityProductLabel() + " · Gumbel · RP " + reportPeriodLabel + " yr");
  const rows = plotCurveRows(params);
  const durations = rows.map((row) => Number(row.duration_min));
  const reportPeriods = activePlotReturnPeriods("city");
  const values = reportPeriods.flatMap((period) => rows.map((row) => Number(idfIntensity(params, Number(row.duration_min), period)))).filter(Number.isFinite);
  const x0 = 76;
  const y0 = 92;
  const plotWidth = 654;
  const plotHeight = 300;
  const maxValue = Math.max(...values, 1) * 1.08;
  const minDuration = Math.min(...durations, GRIDF_PLOT_MIN_DURATION);
  const maxDuration = Math.max(...durations, 1440);
  const xPosition = (duration) => x0 + (Math.log(duration) - Math.log(minDuration)) / (Math.log(maxDuration) - Math.log(minDuration) || 1) * plotWidth;
  const yPosition = (value) => y0 + plotHeight - (value / maxValue) * plotHeight;
  context.fillStyle = "#edf3f0";
  context.fillRect(x0, y0, plotWidth, plotHeight);
  context.strokeStyle = "#cbd9d5";
  context.lineWidth = 1;
  context.fillStyle = "#647b82";
  context.font = "400 11px DM Sans, Arial, sans-serif";
  [0, .25, .5, .75, 1].forEach((fraction) => {
    const y = y0 + plotHeight - fraction * plotHeight;
    context.beginPath(); context.moveTo(x0, y); context.lineTo(x0 + plotWidth, y); context.stroke();
    context.fillText(Math.round(maxValue * fraction).toLocaleString(cityReportLocale()), 18, y + 4);
  });
  [15, 30, 60, 360, 1440].forEach((duration) => {
    const x = xPosition(duration);
    context.beginPath(); context.moveTo(x, y0); context.lineTo(x, y0 + plotHeight); context.stroke();
    context.fillText(formatDuration(duration), x - 15, y0 + plotHeight + 20);
  });
  context.strokeStyle = "#16333f";
  context.lineWidth = 1.4;
  context.beginPath(); context.moveTo(x0, y0); context.lineTo(x0, y0 + plotHeight); context.lineTo(x0 + plotWidth, y0 + plotHeight); context.stroke();
  context.save();
  context.translate(15, y0 + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillStyle = "#16333f";
  context.font = "600 12px DM Sans, Arial, sans-serif";
  context.fillText(state.lang === "pt" ? "Intensidade (mm/h)" : "Intensity (mm/h)", 0, 0);
  context.restore();
  context.fillStyle = "#16333f";
  context.font = "600 12px DM Sans, Arial, sans-serif";
  const xLabel = state.lang === "pt" ? "Duração (escala logarítmica)" : "Duration (logarithmic scale)";
  context.fillText(xLabel, x0 + plotWidth / 2 - context.measureText(xLabel).width / 2, y0 + plotHeight + 43);
  reportPeriods.forEach((period, seriesIndex) => {
    const style = styleForReturnPeriod(period, seriesIndex);
    context.strokeStyle = style.color;
    context.fillStyle = style.color;
    context.lineWidth = 2.2;
    context.beginPath();
    let started = false;
    rows.forEach((row) => {
      const value = Number(idfIntensity(params, Number(row.duration_min), period));
      if (!Number.isFinite(value)) return;
      const x = xPosition(Number(row.duration_min));
      const y = yPosition(value);
      if (!started) { context.moveTo(x, y); started = true; } else context.lineTo(x, y);
    });
    context.stroke();
    rows.forEach((row) => {
      const value = Number(idfIntensity(params, Number(row.duration_min), period));
      if (!Number.isFinite(value)) return;
      drawIdfCanvasMarker(context, xPosition(Number(row.duration_min)), yPosition(value), style.marker, style.color, 2.9);
    });
    const legendX = 82 + (seriesIndex % 5) * 124;
    const legendY = 72 + Math.floor(seriesIndex / 5) * 16;
    context.fillRect(legendX, legendY - 7, 24, 3);
    drawIdfCanvasMarker(context, legendX + 12, legendY - 6, style.marker, style.color, 2.6);
    context.fillStyle = "#16333f";
    context.font = "400 10px DM Sans, Arial, sans-serif";
    context.fillText(plotReturnPeriodLabel(period), legendX + 30, legendY);
  });
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
};

cityReportMapFigure = function (record) {
  const scale = 5;
  const width = 780;
  const height = 460;
  const canvas = document.createElement("canvas");
  canvas.width = width * scale;
  canvas.height = height * scale;
  const context = canvas.getContext("2d");
  context.scale(scale, scale);
  const mapSubtitle = `${record.name} · ${cityProductLabel()} · Gumbel · RP ${state.cityReturnPeriod} yr · ${formatDuration(state.cityDuration)}`;
  cityReportDrawHeader(context, width, t("cityReportFigureMap"), mapSubtitle);
  const nationalBounds = [-74.2, -34.2, -28.4, 5.7];
  const feature = state.data.cities?.features?.find((item) => String(item.properties?.code) === String(record.code));
  const records = cityRecords();
  const scaleValues = cityScale(records);
  context.fillStyle = "#edf3f0";
  context.fillRect(32, 82, 330, 305);
  (state.data.cities?.features || []).forEach((item) => {
    const itemRecord = cityRecord(item.properties?.code);
    const value = itemRecord ? cityIntensity(itemRecord) : null;
    const fill = Number.isFinite(Number(value)) ? cityColor(value, scaleValues.min, scaleValues.max) : "#d4d4d4";
    cityReportDrawGeometry(context, item, nationalBounds, [32, 82, 330, 305], fill, "#ffffff", .18);
  });
  cityReportDrawGeometry(context, state.data.brazil, nationalBounds, [32, 82, 330, 305], null, "#16333f", 1.15);
  cityReportDrawGeometry(context, feature, nationalBounds, [32, 82, 330, 305], "rgba(232,138,50,.72)", "#16333f", 1.6);
  const localBounds = cityReportBounds(feature || state.data.brazil);
  const lonPad = Math.max((localBounds[2] - localBounds[0]) * .18, .01);
  const latPad = Math.max((localBounds[3] - localBounds[1]) * .18, .01);
  const localBox = [420, 86, 288, 255];
  context.fillStyle = "#edf3f0";
  context.fillRect(localBox[0], localBox[1], localBox[2], localBox[3]);
  cityReportDrawGeometry(context, feature, [localBounds[0] - lonPad, localBounds[1] - latPad, localBounds[2] + lonPad, localBounds[3] + latPad], localBox, "#e9ad61", "#16333f", 1.6);
  context.fillStyle = "#16333f";
  context.font = "700 14px DM Sans, Arial, sans-serif";
  context.fillText(record.name, 420, 374);
  context.font = "400 11px DM Sans, Arial, sans-serif";
  context.fillStyle = "#647b82";
  context.fillText(state.lang === "pt" ? "Mapa municipal colorido pela intensidade selecionada" : "Municipal map colored by selected intensity", 32, 412);
  context.fillText(state.lang === "pt" ? "Polígono municipal selecionado" : "Selected municipal polygon", 420, 392);
  context.fillStyle = "#16333f";
  context.font = "700 11px DM Sans, Arial, sans-serif";
  context.fillText(t("cityReportMapLegend") + " · RP " + state.cityReturnPeriod + " yr · " + formatDuration(state.cityDuration), 420, 418);
  const gradient = context.createLinearGradient(420, 432, 708, 432);
  [[0, "#143b5a"], [.25, "#176b78"], [.5, "#55b99f"], [.75, "#f0be62"], [1, "#cf5a53"]].forEach(([stop, color]) => gradient.addColorStop(stop, color));
  context.fillStyle = gradient;
  context.fillRect(420, 430, 288, 10);
  context.fillStyle = "#647b82";
  context.font = "400 10px DM Sans, Arial, sans-serif";
  context.fillText(fmt(scaleValues.min, 1) + " " + t("mmHour"), 420, 454);
  const maxLabel = fmt(scaleValues.max, 1) + " " + t("mmHour");
  context.fillText(maxLabel, 708 - context.measureText(maxLabel).width, 454);
  return pngDataUrlWithDpi(canvas.toDataURL("image/png"), 600);
};

downloadCityReport = async function (record) {
  const params = cityParams(record);
  if (!params || !isFiniteNumber(params.K)) { showToast(t("cityNoData")); return; }
  try {
    const logoFigure = await cityReportLogoFigure();
    const locationFigure = cityReportMapFigure(record);
    const curveFigure = cityReportCurveFigure(record, params);
    const dailyCoefficientFigure = cityReportCoefficientFigure(record, "daily");
    const referenceCoefficientFigure = cityReportCoefficientFigure(record, "reference");
    const productFigure = cityReportProductFigure(record);
    const allRows = curveRows(params);
    const selectedRows = [5, 10, 15, 30, 60, 360, 720, 1440].map((duration) => {
      const intensity = idfIntensity(params, duration, state.cityReturnPeriod);
      return [formatDuration(duration), fmt(intensity, 2), fmt(intensity == null ? null : intensity * duration / 60, 2)];
    });
    const disaggRows = cityDisaggRows(record);
    const disaggTable = [[t("duration"), t("cityDailyCoefficient"), t("cityReferenceCoefficient"), t("citySubdailyReference")]].concat(disaggRows.map((row) => {
      const entry = cityReportDisaggMeta("relative_to_subdaily", row.duration);
      return [formatDuration(row.duration), fmt(row.daily, 4), fmt(row.reference, 4), state.lang === "pt" ? entry?.referenceLabelPt || "" : entry?.referenceLabel || ""];
    }));
    const productTable = [[t("dataset"), t("intensity") + " (" + t("mmHour") + ")", "K", "a", "b", "c"]].concat(cityReportProductRows(record).map((row) => [row.product, fmt(row.intensity, 2), fmt(row.params.K, 2), fmt(row.params.a, 4), fmt(row.params.b, 3), fmt(row.params.c, 4)]));
    const parametersTable = [
      [t("parameters"), state.lang === "pt" ? "Valor" : "Value"],
      ["K", fmt(params.K, 3)], ["a", fmt(params.a, 4)], ["b", fmt(params.b, 3)], ["c", fmt(params.c, 4)],
      [t("cityDailyDepth"), fmt(params.q24?.[String(state.cityReturnPeriod)], 2) + " mm"],
      [t("cityDailyCoefficient"), fmt(cityCoefficient(record), 4)], [t("cityReferenceCoefficient"), fmt(cityReferenceCoefficient(record), 4)], ["R2", fmt(params.R2, 4)],
      [t("cityYears"), fmt(params.Nyears, 0)], [t("cityCoverage"), fmt(Number(params.validAreaFractionMean) * 100, 1) + "%"], [t("citySupport"), `${fmt(params.validPixels, 0)} / ${fmt(params.touchedCells, 0)}`], [t("cityArea"), fmt(record.areaKm2, 2) + " km2"]
    ];
    const designTable = [[t("duration"), t("intensity") + " (" + t("mmHour") + ")", t("depth") + " (" + t("mm") + ")"]].concat(selectedRows);
    const sections = [
      docxImageParagraph("rId2", "GRIDF-BR logo", 2.7, .675, 1),
      docParagraph("GRIDF-BR · Municipal IDFs", "Eyebrow"),
      docParagraph(t("cityReportTitle") + " — " + record.name, "Title"),
      docParagraph(t("cityReportSubtitle"), "Subtitle"),
      docHeading(t("cityReportLocation")),
      docParagraph(record.name + ", " + record.state + " (" + record.stateCode + ") · " + fmt(record.latitude, 4) + ", " + fmt(record.longitude, 4)),
      docParagraph(cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel · " + t("biasLabel") + " · " + t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + t("duration") + ": " + formatDuration(state.cityDuration)),
      docxImageParagraph("rId3", "Municipal intensity map", 6.3, 3.72, 2),
      docFigureCaption(1, cityReportMapCaption(record)),
      docHeading(t("cityReportParameters")),
      docTableCaption(1, cityReportParametersCaption(record)),
      docTable(parametersTable),
      docHeading(t("cityReportMethod")),
      docParagraph(t("cityReportTheory")),
      docHeading(t("cityReportEquationHeading"), "Heading2"),
      docEquationParagraph(),
      docMixedParagraph([{text: state.lang === "pt" ? "Na equação, " : "In the equation, "}, {text: "I", math: true}, {text: state.lang === "pt" ? " é a intensidade, " : " is rainfall intensity, "}, {text: "RP", math: true}, {text: state.lang === "pt" ? " é o período de retorno, " : " is return period, "}, {text: "t", math: true}, {text: state.lang === "pt" ? " é a duração em minutos, e " : " is duration in minutes, and "}, {text: "K, a, b, c", math: true}, {text: state.lang === "pt" ? " são os parâmetros ajustados." : " are fitted parameters."}]),
      docParagraph(t("cityReportMunicipal")),
      docParagraph(t("cityReportWorkflow")),
      docHeading(t("cityReportFrequency")),
      docParagraph(t("cityReportFrequencyText")),
      docParagraph(t("cityReportNoAnnualSeries"), "Small"),
      docxImageParagraph("rId4", "IDF curves", 6.3, 3.88, 3),
      docFigureCaption(2, cityReportCurveCaption(record)),
      docHeading(t("cityReportProductSubheading"), "Heading2"),
      docxImageParagraph("rId7", "Rainfall product comparison", 6.3, 3.15, 4),
      docFigureCaption(3, cityReportProductCaption(record)),
      docTableCaption(2, cityReportProductTableCaption(record)),
      docTable(productTable),
      docSectionBreak(),
      docHeading(t("cityReportDisaggSubheading")),
      docParagraph(t("cityReportDisaggNote")),
      docTableCaption(3, cityReportDisaggregationTableCaption(record)),
      docTable(disaggTable),
      docxImageParagraph("rId5", "Daily disaggregation coefficients", 6.3, 3.47, 5),
      docFigureCaption(4, cityReportDailyCoefficientCaption(record)),
      docSectionBreak(),
      docHeading(t("cityReportDesignSubheading")),
      docParagraph(t("returnPeriod") + ": " + state.cityReturnPeriod + " yr · " + cityProductLabel() + " · " + cityMethodLabel() + " · Gumbel"),
      docTableCaption(4, cityReportDesignTableCaption(record)),
      docTable(designTable),
      docHeading(t("limitsTitle")),
      docParagraph(t("cityReportInterpretation")),
      "<w:sectPr><w:footerReference w:type=\"default\" r:id=\"rId8\"/><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"720\" w:right=\"720\" w:bottom=\"720\" w:left=\"720\"/></w:sectPr>"
    ].join("");
    const documentXml = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\"><w:body>" + sections + "</w:body></w:document>";
    const styles = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:style w:type=\"paragraph\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/><w:pPr><w:spacing w:after=\"145\" w:line=\"285\" w:lineRule=\"auto\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\" w:cs=\"DM Sans\"/><w:color w:val=\"16333F\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Eyebrow\"><w:name w:val=\"Eyebrow\"/><w:pPr><w:spacing w:before=\"120\" w:after=\"60\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"18\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Title\"><w:name w:val=\"Title\"/><w:pPr><w:keepNext/><w:spacing w:before=\"180\" w:after=\"100\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Fraunces\" w:hAnsi=\"Fraunces\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"38\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Subtitle\"><w:name w:val=\"Subtitle\"/><w:pPr><w:spacing w:after=\"140\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"21\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Small\"><w:name w:val=\"Small\"/><w:pPr><w:spacing w:after=\"90\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:color w:val=\"647B82\"/><w:sz w:val=\"17\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Caption\"><w:name w:val=\"Caption\"/><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"45\" w:after=\"180\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"16\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"TableCaption\"><w:name w:val=\"Table Caption\"/><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"150\" w:after=\"70\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"17\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Heading1\"><w:name w:val=\"Heading 1\"/><w:pPr><w:keepNext/><w:spacing w:before=\"300\" w:after=\"120\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"Fraunces\" w:hAnsi=\"Fraunces\"/><w:b/><w:color w:val=\"176B78\"/><w:sz w:val=\"26\"/></w:rPr></w:style><w:style w:type=\"paragraph\" w:styleId=\"Heading2\"><w:name w:val=\"Heading 2\"/><w:pPr><w:keepNext/><w:spacing w:before=\"220\" w:after=\"90\"/></w:pPr><w:rPr><w:rFonts w:ascii=\"DM Sans\" w:hAnsi=\"DM Sans\"/><w:b/><w:color w:val=\"16333F\"/><w:sz w:val=\"21\"/></w:rPr></w:style></w:styles>";
    const stylesWithDefaults = styles.replace('<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">', '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Avenir Next" w:hAnsi="Avenir Next" w:cs="Avenir Next"/><w:color w:val="16333F"/><w:sz w:val="21"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="145" w:line="285" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>').replaceAll("DM Sans", "Avenir Next").replaceAll("Fraunces", "Avenir Next");
    const zip = new JSZip();
    zip.file("[Content_Types].xml", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/><Default Extension=\"png\" ContentType=\"image/png\"/><Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/><Override PartName=\"/word/styles.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/><Override PartName=\"/word/footer1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml\"/></Types>");
    zip.folder("_rels").file(".rels", "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/></Relationships>");
    zip.folder("word").file("document.xml", documentXml);
    zip.folder("word").file("styles.xml", stylesWithDefaults);
    zip.folder("word").file("footer1.xml", docxFooterXml());
    zip.folder("word").folder("_rels").file("document.xml.rels", docxImageRelationships());
    zip.folder("word").folder("media").file("gridf-logo.png", dataUrlBytes(logoFigure));
    zip.folder("word").folder("media").file("city-location.png", dataUrlBytes(locationFigure));
    zip.folder("word").folder("media").file("city-idf-curves.png", dataUrlBytes(curveFigure));
    zip.folder("word").folder("media").file("city-coeff-daily.png", dataUrlBytes(dailyCoefficientFigure));
    zip.folder("word").folder("media").file("city-product-comparison.png", dataUrlBytes(productFigure));
    downloadBlob(cityReportFileName(record), await zip.generateAsync({type: "blob", compression: "DEFLATE"}));
    showToast(t("available"));
  } catch (error) {
    console.error(error);
    showToast(t("rasterError"));
  }
};

state.returnPeriod = 10;
state.cityReturnPeriod = 10;
state.plotReturnPeriods = DEFAULT_PLOT_RETURN_PERIODS.slice();
state.cityPlotReturnPeriods = DEFAULT_PLOT_RETURN_PERIODS.slice();

function docxFooterXml() {
  return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><w:ftr xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:p><w:pPr><w:jc w:val=\"center\"/><w:spacing w:before=\"80\"/></w:pPr><w:r><w:rPr><w:rFonts w:ascii=\"Avenir Next\" w:hAnsi=\"Avenir Next\" w:cs=\"Avenir Next\"/><w:i/><w:color w:val=\"647B82\"/><w:sz w:val=\"16\"/></w:rPr><w:t>GRIDF-BR municipal IDF toolbox</w:t></w:r></w:p></w:ftr>";
}

docxImageRelationships = function () {
  return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/gridf-logo.png\"/><Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-location.png\"/><Relationship Id=\"rId4\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-idf-curves.png\"/><Relationship Id=\"rId5\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-coeff-daily.png\"/><Relationship Id=\"rId7\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/city-product-comparison.png\"/><Relationship Id=\"rId8\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer\" Target=\"footer1.xml\"/></Relationships>";
};

Object.assign(COPY.en, {
  plotReturnPeriods: "Return periods in the chart (years)",
  plotReturnPeriodsPlaceholder: "2, 5, 10, 25, 50, 100",
  plotReturnPeriodsInvalid: "Use positive years separated by commas.",
  feedbackTrigger: "Feedback",
  feedbackKicker: "HELP IMPROVE THE TOOL",
  feedbackTitle: "Send a suggestion",
  feedbackIntro: "Your email application will open with your message and the current technical context already included.",
  feedbackNameLabel: "Your name",
  feedbackEmailLabel: "Email for a reply (optional)",
  feedbackMessageLabel: "Suggestion or comment",
  feedbackContextLabel: "CONTEXT INCLUDED",
  feedbackSubmit: "Prepare email",
  cancel: "Cancel",
  feedbackRequired: "Please provide your name and a message.",
  feedbackEmailSubject: "GRIDF-BR feedback",
  feedbackOpened: "Your email message is ready to review and send.",
  feedbackTimestamp: "Browser-local date and time",
  feedbackView: "Current view",
  feedbackProduct: "Rainfall product",
  feedbackMethod: "Disaggregation method",
  feedbackDuration: "Duration",
  feedbackReturnPeriod: "Return period",
  feedbackLayer: "Map layer",
  feedbackMessageHeading: "Message",
  feedbackTechnicalHeading: "Technical context",
  shareTrigger: "Share",
  shareKicker: "SHARE THE TOOL",
  shareTitle: "Share GRIDF-BR",
  shareEmail: "Email",
  shareCopy: "Copy message",
  shareCopied: "Share message copied.",
  shareInstagramCopied: "Message copied. Paste it in Instagram to share.",
  shareSubject: "GRIDF-BR IDF curves for Brazil"
});

Object.assign(COPY.pt, {
  plotReturnPeriods: "Períodos de retorno no gráfico (anos)",
  plotReturnPeriodsPlaceholder: "2, 5, 10, 25, 50, 100",
  plotReturnPeriodsInvalid: "Use anos positivos separados por vírgulas.",
  feedbackTrigger: "Sugestões",
  feedbackKicker: "CONTRIBUA COM A FERRAMENTA",
  feedbackTitle: "Enviar uma sugestão",
  feedbackIntro: "Seu aplicativo de e-mail será aberto com a mensagem e o contexto técnico atual já incluídos.",
  feedbackNameLabel: "Seu nome",
  feedbackEmailLabel: "E-mail para resposta (opcional)",
  feedbackMessageLabel: "Sugestão ou comentário",
  feedbackContextLabel: "CONTEXTO INCLUÍDO",
  feedbackSubmit: "Preparar e-mail",
  cancel: "Cancelar",
  feedbackRequired: "Informe seu nome e uma mensagem.",
  feedbackEmailSubject: "Sugestão sobre o GRIDF-BR",
  feedbackOpened: "A mensagem de e-mail está pronta para você revisar e enviar.",
  feedbackTimestamp: "Data e hora local do navegador",
  feedbackView: "Aba atual",
  feedbackProduct: "Produto de chuva",
  feedbackMethod: "Método de desagregação",
  feedbackDuration: "Duração",
  feedbackReturnPeriod: "Período de retorno",
  feedbackLayer: "Camada do mapa",
  feedbackMessageHeading: "Mensagem",
  feedbackTechnicalHeading: "Contexto técnico",
  shareTrigger: "Compartilhar",
  shareKicker: "DIVULGUE A FERRAMENTA",
  shareTitle: "Compartilhar GRIDF-BR",
  shareEmail: "E-mail",
  shareCopy: "Copiar mensagem",
  shareCopied: "Mensagem de compartilhamento copiada.",
  shareInstagramCopied: "Mensagem copiada. Cole no Instagram para compartilhar.",
  shareSubject: "GRIDF-BR: curvas IDF para o Brasil"
});

Object.assign(COPY.en, {
  brandSubtitle: "Bias-corrected gridded IDF curves for Brazil",
  navAtlas: "0.1° IDFs",
  navCities: "Municipal IDFs",
  atlasTitle: "Bias-corrected gridded IDF curves",
  atlasControls: "IDF controls",
  productNote: "This interface uses bias-corrected parameter stacks. Raw parameter stacks are intentionally excluded from this decision-support interface.",
  disaggIntro: "Explore the spatial coefficient surfaces used to convert daily or reference-duration rainfall into the durations used by the IDF curves.",
  methodsIntro: "The browser tool exposes the bias-corrected gridded IDF curves first, then documents the supporting station and coefficient pathways.",
  workflowText: "Daily annual maxima are adjusted against the daily-rainfall gauge calibration sample, converted to sub-daily durations using the selected disaggregation pathway, and represented by the Sherman parameter stack. Temporal-distribution coefficients are derived from ANA telemetric observations and interpolated with inverse-distance weighting for the gridded surfaces.",
  productText: "The browser bundle includes bias-corrected parameter stacks for BR-DWGD, IMERG, CHIRPS, and PERSIANN-CDR. The interface retains the three disaggregation pathways: local/interpolated, CETESB fixed ratios, and station-derived ratios.",
  stationText: "The disaggregation section displays the station coefficient export and the interpolated coefficient surfaces used by the IDF curves.",
  limitsText: "This is a spatial decision-support product based on gridded model outputs. It does not replace site-specific engineering verification, and the displayed fit diagnostics should not be read as complete uncertainty bands or independent validation of the full IDF chain.",
  welcomeIntro: "Choose a bias-corrected rainfall product, click the IDF map, and download the resulting curve and design values.",
  signalAtlas: "0.1° IDFs",
  atlas: "IDF curves",
  citiesTitle: "Municipal IDFs",
  cityReportFrequencyText: "Municipal IDFs use the Gumbel distribution fitted by the method of moments for all city-scale frequency estimates.",
  cityReportGenerated: "Generated by the GRIDF-BR municipal IDF tool"
});

Object.assign(COPY.pt, {
  brandSubtitle: "Curvas IDF gradeadas corrigidas por viés para o Brasil",
  navAtlas: "0.1° IDFs",
  navCities: "IDFs municipais",
  atlasTitle: "Curvas IDF gradeadas corrigidas por viés",
  atlasControls: "Controles IDF",
  productNote: "Esta interface usa pilhas de parâmetros corrigidas por viés. As pilhas sem correção foram intencionalmente excluídas desta interface de apoio à decisão.",
  disaggIntro: "Explore as superfícies espaciais de coeficientes usadas para converter a chuva diária ou de duração de referência nas durações usadas pelas curvas IDF.",
  methodsIntro: "A ferramenta expõe primeiro as curvas IDF gradeadas corrigidas por viés e depois documenta os caminhos de estações e coeficientes.",
  workflowText: "Os máximos anuais diários são ajustados em relação à amostra de calibração de estações de chuva diária, convertidos para durações subdiárias pelo caminho de desagregação selecionado e representados pela pilha de parâmetros Sherman. Os coeficientes de distribuição temporal são derivados de observações telemétricas da ANA e interpolados por ponderação pelo inverso da distância para as superfícies gradeadas.",
  productText: "O pacote do navegador inclui pilhas de parâmetros corrigidas por viés para BR-DWGD, IMERG, CHIRPS e PERSIANN-CDR. A interface mantém os três caminhos de desagregação: local/interpolado, razões fixas CETESB e derivado de estações.",
  stationText: "A seção de desagregação exibe o exportador de coeficientes das estações e as superfícies interpoladas usadas pelas curvas IDF.",
  limitsText: "Este é um produto espacial de apoio à decisão baseado em resultados de modelos gradeados. Ele não substitui a verificação de engenharia local, e os diagnósticos de ajuste exibidos não devem ser lidos como bandas completas de incerteza ou validação independente de toda a cadeia IDF.",
  welcomeIntro: "Escolha um produto de chuva corrigido por viés, clique no mapa IDF e baixe a curva e os valores de projeto.",
  signalAtlas: "0.1° IDFs",
  atlas: "Curvas IDF",
  citiesTitle: "IDFs municipais",
  cityReportFrequencyText: "As IDFs municipais utilizam a distribuição Gumbel ajustada pelo método dos momentos para todas as estimativas de frequência em escala municipal.",
  cityReportGenerated: "Gerado pela ferramenta municipal de IDF do GRIDF-BR"
});

const FEEDBACK_EMAIL = "marcusnobrega.engcivil@gmail.com";

function feedbackContextLines() {
  const timestamp = new Date().toLocaleString(state.lang === "pt" ? "pt-BR" : "en-US", {dateStyle: "medium", timeStyle: "medium"});
  const lines = [
    `${t("feedbackTimestamp")}: ${timestamp}`,
    `${t("feedbackView")}: ${state.view === "cities" ? t("navCities") : state.view === "methods" ? t("navMethods") : t("navAtlas")}`
  ];
  if (state.selected) {
    lines.push(`${t("feedbackProduct")}: ${selectedProductLabel()}`);
    lines.push(`${t("feedbackMethod")}: ${selectedMethodLabel()}`);
    lines.push(`${t("feedbackDuration")}: ${formatDuration(state.duration)}`);
    lines.push(`${t("feedbackReturnPeriod")}: ${state.returnPeriod} yr`);
    lines.push(`${t("feedbackLayer")}: ${LAYER_OPTIONS.find((option) => option.value === state.layer)?.[state.lang] || state.layer}`);
  } else if (state.citySelected) {
    lines.push(`${t("feedbackProduct")}: ${cityProductLabel()}`);
    lines.push(`${t("feedbackMethod")}: ${cityMethodLabel()}`);
    lines.push(`${t("feedbackDuration")}: ${formatDuration(state.cityDuration)}`);
    lines.push(`${t("feedbackReturnPeriod")}: ${state.cityReturnPeriod} yr`);
  }
  return lines;
}

function refreshFeedbackContext() {
  const context = $("feedbackContext");
  const button = $("feedbackButton");
  if (context) context.textContent = feedbackContextLines().join("\n");
  if (button) {
    const label = t("feedbackTrigger");
    button.title = label;
    button.setAttribute("aria-label", label);
  }
}

function closeFeedback() {
  $("feedbackModal").hidden = true;
}

function openFeedback() {
  refreshFeedbackContext();
  $("feedbackModal").hidden = false;
  $("feedbackName").focus();
}

function submitFeedback(event) {
  event.preventDefault();
  const name = $("feedbackName").value.trim();
  const email = $("feedbackEmail").value.trim();
  const message = $("feedbackMessage").value.trim();
  if (!name || !message) {
    showToast(t("feedbackRequired"));
    return;
  }
  const messageLines = [
    `${t("feedbackMessageHeading")}:`,
    message,
    "",
    `${t("feedbackNameLabel")}: ${name}`,
    email ? `${t("feedbackEmailLabel")}: ${email}` : "",
    "",
    `${t("feedbackTechnicalHeading")}:`,
    ...feedbackContextLines(),
    "",
    `URL: ${window.location.href}`
  ].filter(Boolean);
  const subject = `${t("feedbackEmailSubject")} - ${name}`;
  window.location.href = `mailto:${FEEDBACK_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(messageLines.join("\n"))}`;
  closeFeedback();
  $("feedbackForm").reset();
  showToast(t("feedbackOpened"));
}

function currentVisualizationUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("v", "idf-naming-20260723");
  url.searchParams.set("view", state.view);
  if (state.view === "cities") {
    url.searchParams.set("product", state.cityProduct || state.product);
    url.searchParams.set("method", state.cityMethod || "local-interpolated");
    url.searchParams.set("duration", state.cityDuration);
    url.searchParams.set("return_period", state.cityReturnPeriod);
    if (state.citySelected?.code) url.searchParams.set("city", state.citySelected.code);
  } else {
    url.searchParams.set("product", state.product);
    url.searchParams.set("method", state.method);
    url.searchParams.set("duration", state.duration);
    url.searchParams.set("return_period", state.returnPeriod);
    url.searchParams.set("layer", state.layer);
    if (state.selected) {
      url.searchParams.set("lat", Number(state.selected.lat).toFixed(5));
      url.searchParams.set("lon", Number(state.selected.lon).toFixed(5));
    }
  }
  return url.toString();
}

function shareBaseMessage() {
  if (state.lang === "pt") {
    return "GRIDF-BR: Curvas IDF em todo o Brasil (0.1°) ou em nível de cidade. Coeficientes de desagregação computados localmente com estações de alta resolução ou pelos propostos pela CETESB.";
  }
  return "GRIDF-BR: IDF curves across Brazil (0.1°) or at city scale. Disaggregation coefficients are computed locally with high-resolution stations or with the coefficients proposed by CETESB.";
}

function shareMessage() {
  return `${shareBaseMessage()}\n\n${currentVisualizationUrl()}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (match) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"}[match]));
}

function shareMessageHtml() {
  const url = currentVisualizationUrl();
  return `${escapeHtml(shareBaseMessage())}<br><br><a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(url)}</a>`;
}

function refreshSharePreview() {
  const preview = $("sharePreview");
  const button = $("shareButton");
  if (preview) preview.innerHTML = shareMessageHtml();
  if (button) {
    const label = t("shareTrigger");
    button.title = label;
    button.setAttribute("aria-label", label);
  }
}

async function copyShareMessage(silent = false) {
  const text = shareMessage();
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.left = "-9999px";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    if (!silent) showToast(t("shareCopied"));
  } catch (error) {
    console.error(error);
    showToast(t("rasterError"));
  }
}

function closeShare() {
  $("shareModal").hidden = true;
}

function openShare() {
  refreshSharePreview();
  $("shareModal").hidden = false;
  $("copyShareButton").focus();
}

function openShareUrl(url) {
  window.open(url, "_blank", "noopener,noreferrer");
}

function initializeShare() {
  $("shareButton").addEventListener("click", openShare);
  $("closeShareButton").addEventListener("click", closeShare);
  $("shareModal").addEventListener("click", (event) => { if (event.target === $("shareModal")) closeShare(); });
  $("copyShareButton").addEventListener("click", () => copyShareMessage());
  $("shareWhatsappButton").addEventListener("click", () => openShareUrl(`https://wa.me/?text=${encodeURIComponent(shareMessage())}`));
  $("shareFacebookButton").addEventListener("click", () => openShareUrl(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(currentVisualizationUrl())}&quote=${encodeURIComponent(shareBaseMessage())}`));
  $("shareEmailButton").addEventListener("click", () => { window.location.href = `mailto:?subject=${encodeURIComponent(t("shareSubject"))}&body=${encodeURIComponent(shareMessage())}`; });
  $("shareInstagramButton").addEventListener("click", async () => {
    await copyShareMessage(true);
    showToast(t("shareInstagramCopied"));
    openShareUrl("https://www.instagram.com/");
  });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !$("shareModal").hidden) closeShare(); });
  refreshSharePreview();
  refreshIcons();
}

function initializeFeedback() {
  $("feedbackButton").addEventListener("click", openFeedback);
  $("closeFeedbackButton").addEventListener("click", closeFeedback);
  $("cancelFeedbackButton").addEventListener("click", closeFeedback);
  $("feedbackModal").addEventListener("click", (event) => { if (event.target === $("feedbackModal")) closeFeedback(); });
  $("feedbackForm").addEventListener("submit", submitFeedback);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !$("feedbackModal").hidden) closeFeedback(); });
  refreshFeedbackContext();
  refreshIcons();
}

const setLanguageWithFeedback = setLanguage;
setLanguage = function (language) {
  setLanguageWithFeedback(language);
  refreshFeedbackContext();
  refreshSharePreview();
};

initializeShare();
initializeFeedback();
setLanguage(state.lang);
