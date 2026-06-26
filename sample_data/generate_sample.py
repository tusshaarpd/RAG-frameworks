"""Generates the bundled 3-page sample PDF about crop protection basics.

Run once:  python sample_data/generate_sample.py
"""
from pathlib import Path

from fpdf import FPDF

PAGES = [
    ("Crop Protection Basics - Page 1: Foundations", """
Crop protection is the practice of defending cultivated plants against
yield losses caused by pests, diseases, and weeds. Globally, an estimated
20 to 40 percent of potential crop production is lost to these threats
each year, making crop protection one of the highest-leverage activities
in agriculture.

The three principal categories of crop threats are: insect pests such as
aphids, stem borers, and locusts; plant diseases caused by fungi,
bacteria, and viruses; and weeds, which compete with crops for light,
water, and nutrients. Each category demands different detection methods
and different interventions.

Integrated Pest Management, commonly abbreviated IPM, is an ecosystem-based
strategy that focuses on long-term prevention of pests through a
combination of techniques: biological control, habitat manipulation,
modification of cultural practices, and the use of resistant crop
varieties. Under IPM, chemical pesticides are used only after monitoring
indicates they are needed according to established thresholds, and
treatments are chosen to minimize risks to human health, beneficial
organisms, and the environment.
"""),
    ("Crop Protection Basics - Page 2: Methods", """
Biological control uses living organisms to suppress pest populations.
Classic examples include releasing ladybird beetles to control aphids and
applying Bacillus thuringiensis, a soil bacterium, against caterpillar
pests. Biological control is slower than chemical control but far more
durable and leaves no residues.

Cultural control modifies the farming environment to make it less
hospitable to pests. Crop rotation breaks pest life cycles, intercropping
confuses host-seeking insects, and adjusting planting dates can help a
crop escape peak pest pressure. Sanitation, such as removing crop residues
that harbor overwintering insects, is among the cheapest and most
effective cultural measures.

Chemical control relies on pesticides: insecticides, fungicides, and
herbicides. Within IPM, pesticides are a last resort, applied only when
pest populations exceed the economic threshold - the point at which the
cost of damage exceeds the cost of treatment. Rotating pesticide modes of
action is essential to delay the evolution of resistance.
"""),
    ("Crop Protection Basics - Page 3: Monitoring and the Future", """
Monitoring, also called scouting, is the backbone of IPM. Farmers inspect
fields on a regular schedule, count pests using traps such as pheromone
traps and sticky cards, and compare counts against published thresholds.
Modern scouting increasingly uses drone imagery and smartphone
identification apps to detect problems earlier than the human eye can.

Economic thresholds vary by crop and pest. For example, a common rule for
cereal aphids is to treat only when populations exceed roughly five aphids
per tiller before flowering. Spraying below the threshold wastes money and
needlessly harms beneficial insects such as parasitoid wasps and hoverfly
larvae, which often keep aphid populations in check at no cost.

The future of crop protection is precision agriculture: variable-rate
sprayers that treat only infested patches, disease-forecasting models
driven by weather data, gene-edited crop varieties with durable
resistance, and biopesticides derived from natural organisms. These tools
aim to keep yields high while shrinking the environmental footprint of
farming.
"""),
]


def main():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    for title, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", size=11)
        for para in body.strip().split("\n\n"):
            pdf.multi_cell(0, 6, " ".join(para.split()))
            pdf.ln(3)
    out = Path(__file__).parent / "sample.pdf"
    pdf.output(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
