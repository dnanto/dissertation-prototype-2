#!/usr/bin/env python3

import re
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType
from collections import OrderedDict
from datetime import datetime as dt
from operator import itemgetter
from pathlib import Path

from Bio import SeqIO
from bs4 import BeautifulSoup

PATH_TEMPLATES = Path(__file__).parent / "templates"


def parse_log(stream):
    fields = []
    for line in map(str.strip, stream):
        if line.startswith("ModelFinder"):
            fields = next(stream).strip().split()
        if fields:
            tokens = line.split()
            if tokens[0].isdigit():
                yield OrderedDict(zip(fields, tokens))
            elif tokens[0] == "Akaike":
                break


def to_decimal_date(date):
    # https://stackoverflow.com/a/6451892
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def taxa_tags(soup, path):
    # taxa
    tag_tax = soup.new_tag("taxa", id="taxa")
    # alignment
    tag_aln = soup.new_tag("alignment", id="alignment", dataType="nucleotide")

    for rec in SeqIO.parse(path, "fasta"):
        # taxon
        value = to_decimal_date(
            dt.strptime(re.search(r"(\d{4}-\d{2}-\d{2})", rec.id).group(1), "%Y-%m-%d")
        )
        tag_txn = soup.new_tag("taxon", id=rec.id)
        tag_txn.append(soup.new_tag("date", value=str(value), direction="forwards", units="years"))
        tag_tax.append(tag_txn)
        # sequence
        tag_seq = soup.new_tag("sequence")
        tag_seq.append(soup.new_tag("taxon", idref=rec.id))
        tag_seq.append(str(rec.seq).upper())
        tag_aln.append(tag_seq)

    return tag_tax, tag_aln


def gammaize(soup):
    soup.beast.siteModel.append(
        BeautifulSoup(
            """
        <gammaShape gammaCategories="4">
            <parameter id="sitemodel.alpha" value="0.5" lower="0.0"/>
        </gammaShape>
    """,
            "lxml-xml",
        )
    )
    soup.beast.operators.append(
        BeautifulSoup(
            """
        <scaleOperator scaleFactor="0.75" weight="0.1">
            <parameter idref="sitemodel.alpha"/>
        </scaleOperator>
    """,
            "lxml-xml",
        )
    )
    soup.beast.mcmc.joint.prior.append(
        BeautifulSoup(
            """
        <exponentialPrior mean="0.5" offset="0.0">
            <parameter idref="sitemodel.alpha"/>
        </exponentialPrior>
    """,
            "lxml-xml",
        )
    )
    soup.select_one("#fileLog").append(soup.new_tag("parameter", idref="sitemodel.alpha"))


def invariantize(soup):
    soup.beast.siteModel.append(
        BeautifulSoup(
            """
        <proportionInvariant>
            <parameter id="sitemodel.pInv" value="0.5" lower="0.0" upper="1.0"/>
        </proportionInvariant>
    """,
            "lxml-xml",
        )
    )
    soup.beast.operators.append(
        BeautifulSoup(
            """
		<randomWalkOperator windowSize="0.75" weight="1" boundaryCondition="logit">
			<parameter idref="sitemodel.pInv"/>
		</randomWalkOperator>
    """,
            "lxml-xml",
        )
    )
    soup.beast.mcmc.joint.prior.append(
        BeautifulSoup(
            """
        <uniformPrior lower="0.0" upper="1.0">
            <parameter idref="sitemodel.pInv"/>
        </uniformPrior>
    """,
            "lxml-xml",
        )
    )
    soup.select_one("#fileLog").append(soup.new_tag("parameter", idref="sitemodel.pInv"))


def psss_tags(soup, path, **kwargs):
    with open(path) as stream:
        for ele in list(BeautifulSoup(stream, "xml").select_one("mle").children):
            soup.beast.append(ele)
        stem = kwargs["stem"]
        soup.select_one("marginalLikelihoodEstimator")["chainLength"] = kwargs["mle_len"]
        soup.select_one("marginalLikelihoodEstimator")["pathSteps"] = kwargs["mle_step"]
        soup.select_one("#MLELog")["logEvery"] = kwargs["mle_echo"]
        soup.select_one("#MLELog")["fileName"] = stem + ".mle.log"
        soup.select_one("pathSamplingAnalysis")["fileName"] = stem + ".mle.log"
        soup.select_one("pathSamplingAnalysis")["resultsFileName"] = stem + ".mle.result.log"
        soup.select_one("steppingStoneSamplingAnalysis")["fileName"] = stem + ".mle.log"
        soup.select_one("steppingStoneSamplingAnalysis")["resultsFileName"] = (
            stem + ".mle.result.log"
        )


params = snakemake.params

with open(str(snakemake.input.log)) as stream:
    model_bic = (
        min(parse_log(stream), key=itemgetter("BIC"))["Model"]
        .replace("+G4", "+G")
        .replace("+F", "")
    )

with PATH_TEMPLATES.joinpath("model.xml").open() as stream:
    model_id = model_bic.split("+", maxsplit=1)[0]
    model = BeautifulSoup(stream, "xml").find("model", id=model_id)
    sub_model = model.select_one("subModel")
    site_model = model.select_one("#sitemodel")
    operators = model.select_one("operators")
    prior = model.select_one("prior")
    log = model.select_one("log")


with PATH_TEMPLATES.joinpath(f"{params.clock}-{params.coal}.xml").open() as stream:
    soup = BeautifulSoup(stream, "xml")
    # taxa
    tag_tax, tag_aln = taxa_tags(soup, str(snakemake.input.fas))
    soup.beast.insert(0, tag_tax)
    soup.beast.insert(1, tag_aln)
    # model
    soup.beast.insert(2, sub_model)
    soup.beast.insert(3, site_model)
    if model.has_attr("operators"):
        for ele in list(operators.children):
            soup.beast.operators.append(ele)
    if model.has_attr("prior"):
        for ele in list(prior.children):
            soup.beast.mcmc.joint.prior.append(ele)
    if model.has_attr("log"):
        for ele in list(log.children):
            soup.select_one("#fileLog").append(ele)
    if "+G" in model_bic:
        gammaize(soup)
    if "+I" in model_bic:
        invariantize(soup)
    # MCMC
    soup.select_one("mcmc")["chainLength"] = params.mcmc_len
    soup.select_one("mcmc")["operatorAnalysis"] = params.stem + ".ops"
    soup.select_one("#fileLog")["logEvery"] = params.mcmc_echo
    soup.select_one("#fileLog")["fileName"] = params.stem + ".log"
    soup.select_one("logTree")["logEvery"] = params.mcmc_echo
    soup.select_one("logTree")["fileName"] = params.stem + ".trees"
    soup.select_one("#screenLog")["logEvery"] = 0
    # PS/SS
    psss_tags(soup, PATH_TEMPLATES.joinpath("psss.xml"), **vars(params))

with open(snakemake.output.xml, "w") as stream:
    print(soup.prettify(), file=stream)
