      "use strict";
      var BEST_URL = new URL("best.txt", window.location.href).href;
      var OPENCODE_PREFIX = "opencode --model ";
      var CURL_COMMAND = OPENCODE_PREFIX + '"$(curl -fsSL ' + BEST_URL + ')"';

      var els = {
        tbody: document.getElementById("tbody"),
        status: document.getElementById("status"),
        generated: document.getElementById("generated"),
        generatedTop: document.getElementById("generated-top"),
        revBlock: document.getElementById("rev-block"),
        revLink: document.getElementById("rev-link"),
        copyModel: document.getElementById("copy-model"),
        copyCommand: document.getElementById("copy-command"),
        copyCurl: document.getElementById("copy-curl"),
        modelValue: document.getElementById("model-value"),
        commandValue: document.getElementById("command-value"),
        aliasHint: document.getElementById("alias-hint"),
        curlValue: document.getElementById("curl-value"),
      };
      var data = null;
      var historyData = null;
      var catalogData = null;
      var endpointExpanded = {};
      // Artifact failures compose into one status line (renderStatus) so
      // the catalog catch can't clobber a data.json error.
      var dataError = null;
      var catalogError = false;
      var current = "balanced";
      // Table layout: "expert" (default, all 13 columns) or "simple" (the
      // 8-column summary). Persisted via the consent-first preference
      // cookie below.
      var view = "expert";
      // Consent-first: the preference cookie is only written after the
      // user accepts in the dialog. Reject keeps the view for this page
      // load but stores nothing, and it suppresses re-asking only until
      // the next page load. Nothing else is ever stored.
      var COOKIE_NAME = "model-compare-view";
      var consentRejected = false;

      function readCookie(name) {
        var parts = document.cookie.split("; ");
        for (var i = 0; i < parts.length; i++) {
          var kv = parts[i].split("=");
          if (kv[0] === name) {
            // An empty value carries no consent. (A real browser drops
            // the cookie entirely after the max-age=0 deletion; the
            // retained "model-compare-view=" string is what the test
            // stub reports.)
            var value = kv.slice(1).join("=");
            return value === "" ? null : value;
          }
        }
        return null;
      }

      // Attributes shared by the write and the delete, so deletion
      // always matches the write (a mismatch silently breaks removal).
      // The path is scoped to this page's directory: canonical.github.io
      // serves every project page from one origin, and a path=/ cookie
      // with a generic name would leak to — and collide with — sibling
      // projects. Secure is added only where the browser reports a
      // secure context (the https deploy; localhost previews count as
      // secure, a plain-http LAN preview does not and must keep
      // working).
      function cookiePath() {
        return new URL(".", window.location.href).pathname;
      }
      function cookieAttrs() {
        return (
          "; SameSite=Lax; path=" +
          cookiePath() +
          (window.isSecureContext ? "; Secure" : "")
        );
      }
      function writeViewCookie() {
        // One year: a layout preference, not a session.
        document.cookie =
          COOKIE_NAME + "=simple; max-age=31536000" + cookieAttrs();
      }
      function clearViewCookie() {
        document.cookie = COOKIE_NAME + "=; max-age=0" + cookieAttrs();
      }

      // The dialog asks only when no consent decision is stored: not
      // accepted (no cookie) and not rejected on this page load. After
      // OK -> Expert (cookie deleted) -> Simple it asks again, matching
      // the consent-first contract.
      function shouldAskForConsent() {
        return readCookie(COOKIE_NAME) !== "simple" && !consentRejected;
      }

      function applyView() {
        var simple = view === "simple";
        document.body.classList.toggle("view-simple", simple);
        var simpleBtn = document.getElementById("view-simple");
        var expertBtn = document.getElementById("view-expert");
        simpleBtn.classList.toggle("active", simple);
        expertBtn.classList.toggle("active", !simple);
        // The toggle state is announced, not just styled.
        simpleBtn.setAttribute("aria-pressed", String(simple));
        expertBtn.setAttribute("aria-pressed", String(!simple));
      }

      function setView(next) {
        view = next;
        applyView();
        if (data) render();
      }

      // NOTE: fmtPrice/fmtContext/fmtAge mirror the fmt_* helpers in
      // model_compare.py -- display-only duplication, keep them in sync.
      // Tie rounding is half-to-even, like Python's f-strings (toFixed
      // rounds exact ties half-away-from-zero; pyFixed corrects that).
      function pyFixed(v, n) {
        // Python's f"{v:.Nf}" rounds exact decimal ties to even; toFixed
        // rounds them half-away. The CLI's fmt_* helpers are the reference,
        // so match Python: when v sits exactly on an N-digit tie, step the
        // last formatted digit down to the even neighbour. Data arrives
        // rounded to <= 6 decimals, which is always >= 1e-6 away from a
        // non-exact tie, so a 1e-9 absolute tie tolerance cannot misfire.
        var s = v.toFixed(n);
        var tie = Math.abs(v - (Number(s) - Math.pow(10, -n) / 2)) < 1e-9;
        if (tie) {
          var last = parseInt(s.charAt(s.length - 1), 10);
          if (last % 2 === 1) {
            // No borrow is possible: a tie's rounded-up last digit steps
            // down to the even neighbour below ("10.3"->"10.2",
            // "0.063"->"0.062", "101"->"100").
            s = s.slice(0, -1) + (last - 1);
          }
        }
        return s;
      }

      function fmtPrice(v) {
        if (v === 0) return "0";
        if (v >= 100) return pyFixed(v, 0);
        if (v >= 10) return pyFixed(v, 1);
        if (v >= 1) return pyFixed(v, 2);
        return pyFixed(v, 3);
      }

      function fmtContext(v) {
        if (v >= 1e6) return pyFixed(v / 1e6, 2) + "M";
        if (v >= 1e3) return pyFixed(v / 1e3, 0) + "K";
        return String(v);
      }

      function fmtAge(d) {
        if (d === null || d === undefined) return "-";
        if (d >= 365) return pyFixed(d / 365, 1) + "y";
        if (d >= 60) return pyFixed(d / 30.44, 0) + "mo";
        return pyFixed(d, 0) + "d";
      }

      // PREC badge: default_precision or "n/a", "!" when the routing risk
      // flag is set. The flag means default OpenRouter routing can land on
      // cheaper sub-default-quantization endpoints, so the ranked price is
      // not the precision you get.
      function fmtPrec(row) {
        var text =
          row.default_precision === null || row.default_precision === undefined
            ? "n/a"
            : String(row.default_precision);
        if (row.risk_flag) text += "!";
        return text;
      }

      // PREC cell tooltip: the CLI-printed quant_note string verbatim
      // (house rule: display strings from data.json are shown as-is).
      function fmtQuantTooltip(row) {
        return row.quant_note || null;
      }

      // TASK$ cell: the session-cost dataset's median cost for this model's
      // task class, USD. Null/undefined/non-finite means the dataset had no
      // row this refresh -- the "—" state, never zero, never blank-silent.
      // No negative guard: fmtPrice has none either, and a negative session
      // cost is a data bug that must surface, not hide.
      function fmtTaskCost(value) {
        if (value === null || value === undefined) return "—";
        if (typeof value !== "number" || !isFinite(value)) return "—";
        return "$" + pyFixed(value, 2);
      }

      // QUAL provenance: one row's quality as_of, rendered the way the CLI
      // footer's _footer_as_of renders it (display-only duplication, keep
      // them in sync). YYYYMMDD stamps (OpenRouter's benchmark date, and
      // catalog rows hydrated from OR rows) render as a plain date
      // ("2026-10-17") for any source -- a date-only stamp never fabricates
      // a time of day. Live AA records carry one fetch timestamp for the
      // whole fetch (checked today), so they show the time of day only
      // ("13:15Z"); other ISO stamps show the full form ("2026-10-23
      // 13:15Z"). The function returns the bare stamp: the caller owns the
      // connective word (the CLI footer's stamp_words -- "checked" for aa,
      // "as of" elsewhere). An unusable stamp returns null and the caller
      // drops the clause, like the CLI footer drops the stamp. Stamps in
      // data.json are UTC, so the UTC getters are exact.
      function qualityAsOf(source, value) {
        if (value === null || value === undefined || value === "") return null;
        var text = String(value);
        if (text.length === 8 && /^\d+$/.test(text)) {
          return text.slice(0, 4) + "-" + text.slice(4, 6) + "-" + text.slice(6);
        }
        var parsed = new Date(text);
        if (isNaN(parsed.getTime())) return null;
        var pad = function (n) {
          return (n < 10 ? "0" : "") + n;
        };
        var clock =
          pad(parsed.getUTCHours()) + ":" + pad(parsed.getUTCMinutes()) + "Z";
        if (source === "aa") return clock;
        return (
          parsed.getUTCFullYear() +
          "-" +
          pad(parsed.getUTCMonth() + 1) +
          "-" +
          pad(parsed.getUTCDate()) +
          " " +
          clock
        );
      }

      function finiteNumber(value) {
        return typeof value === "number" && isFinite(value) ? value : null;
      }

      function effortOrder(effort) {
        var index = EFFORT_ORDER.indexOf(effort);
        return index === -1 ? EFFORT_ORDER.length : index;
      }

      // Catalog v5 keeps the provenance flags but not the producer's
      // is_reasoning bool. A non-reasoning model has no reasoning metadata
      // at all, so its three flags stay null and its effective default is
      // "none"; a single-state reasoning model with reasoning off records
      // default_enabled: false.
      function isNonReasoningBlock(reasoning) {
        if (!reasoning) return false;
        if (reasoning.is_reasoning === false) return true;
        if (reasoning.is_reasoning === true) return false;
        return (
          reasoning.default_effort === "none" &&
          reasoning.default_enabled !== false &&
          reasoning.mandatory !== true &&
          reasoning.supports_max_tokens !== true
        );
      }

      function isSingleStateToggle(reasoning, supported) {
        if (!reasoning || supported.length) return false;
        if (reasoning.mandatory === true) return false;
        return (
          reasoning.is_reasoning === true || reasoning.default_enabled === false
        );
      }

      // depthRows(row): the union of OpenRouter-supported and AA-measured
      // efforts, deepest first, normalized for both the main table (a
      // data.json row with the catalog reasoning block joined in) and the
      // expanded depth table (a catalog entry). A measured record may carry
      // the pre-release draft key quality_index; the normalized field is
      // index. ratio is AA $/task divided by the default-depth record's,
      // null when either side is missing or zero.
      function depthRows(row) {
        if (!row) return [];
        var reasoning = row.reasoning || {};
        var supported = row.supported_efforts;
        if (!Array.isArray(supported) || !supported.length) {
          if (Array.isArray(reasoning.supported_efforts)) {
            supported = reasoning.supported_efforts;
          }
        }
        if (!Array.isArray(supported)) supported = [];
        var depths = Array.isArray(reasoning.depths) ? reasoning.depths : [];
        var measured = {};
        depths.forEach(function (record) {
          if (record && record.effort) measured[record.effort] = record;
        });
        var defaultEffort =
          row.default_effort || reasoning.default_effort || null;
        var selectedEffort = row.quality_effort || null;
        var nonReasoning =
          !supported.length && isNonReasoningBlock(reasoning);
        var singleStateNone = isSingleStateToggle(reasoning, supported);
        var efforts = [];
        var seen = {};
        supported.forEach(function (effort) {
          if (effort && !seen[effort]) {
            seen[effort] = true;
            efforts.push(effort);
          }
        });
        depths.forEach(function (record) {
          if (record && record.effort && !seen[record.effort]) {
            seen[record.effort] = true;
            efforts.push(record.effort);
          }
        });
        efforts.sort(function (a, b) {
          return effortOrder(a) - effortOrder(b);
        });
        var defaultCost = null;
        if (defaultEffort && measured[defaultEffort]) {
          defaultCost = finiteNumber(measured[defaultEffort].cost_per_task_usd);
        }
        return efforts.map(function (effort) {
          var record = measured[effort] || {};
          var index = finiteNumber(
            record.index === undefined ? record.quality_index : record.index,
          );
          var cost = finiteNumber(record.cost_per_task_usd);
          var selectable = false;
          if (effort === "reasoning") {
            // Single-state reasoning row: the bare id is the only form.
            selectable = false;
          } else if (effort === "none" && nonReasoning) {
            // A non-reasoning model's bare id is the only form.
            selectable = false;
          } else if (supported.indexOf(effort) !== -1) {
            selectable = true;
          } else if (effort === "none" && singleStateNone) {
            selectable = true;
          }
          var ratio = null;
          if (cost !== null && cost !== 0 && defaultCost) {
            ratio = cost / defaultCost;
          }
          return {
            effort: effort,
            index: index,
            cost: cost,
            ratio: ratio,
            selectable: selectable,
            selected: effort === selectedEffort,
            default: effort === defaultEffort,
          };
        });
      }

      // DEPTH cell: the effort QUAL was measured at, "*" on the default.
      function fmtEffort(row) {
        if (!row || !row.quality_effort) return "-";
        var reasoning = row.reasoning || {};
        var defaultEffort = row.default_effort || reasoning.default_effort;
        return row.quality_effort === defaultEffort
          ? row.quality_effort + "*"
          : row.quality_effort;
      }

      function fmtDepthRatio(value) {
        if (value === null) return "-";
        return pyFixed(value, value >= 1 ? 1 : 2) + "x";
      }

      function depthTooltip(row) {
        if (!row || !row.quality_effort) return null;
        var reasoning = row.reasoning || {};
        var defaultEffort = row.default_effort || reasoning.default_effort;
        var text = "Reasoning effort the QUAL score was measured at";
        if (row.quality_effort === defaultEffort) {
          text += " (OpenRouter's effective default)";
        }
        return text + ".";
      }

      // A display-only AA reference: the index was measured without a
      // reasoning-effort label, so it cannot fill QUAL. The number is still
      // worth surfacing because AA does rank the model, but the note must
      // say plainly that a selectable effort may score differently.
      function unlabeledNote(record) {
        var index = finiteNumber(record && record.index);
        if (index === null) return null;
        var cost = finiteNumber(record && record.cost_per_task_usd);
        var text = "AA index " + pyFixed(index, 1) + " without an effort label";
        if (cost !== null) text += " (AA $" + pyFixed(cost, 2) + " per task)";
        return (
          text +
          "; reference only, may not be the quality you get at a selectable effort."
        );
      }

      // QUAL cell tooltip: where the quality number came from (the source
      // label verbatim -- display strings from data.json are shown as-is)
      // and when it was checked, the effort it was measured at, plus any
      // drift note for a row the producer flagged as disagreeing with
      // OpenRouter's snapshot by >= 3
      // index points. The connective word mirrors the CLI footer's
      // stamp_words ("checked" for aa, "as of" elsewhere). The note names
      // the row's own OR pair; a flag without a finite check value cannot
      // name what it disagrees with, so the note is withheld rather than
      // leaking "NaN". A reasoning row's OR snapshot has no effort label,
      // so its reference is not a like-for-like drift check. Rows without
      // provenance (older data.json files, unmatched models) return null,
      // unless AA ranked the model without an effort label: that reference
      // explains why the cell is blank.
      function qualityTooltip(row) {
        var source = row.quality_source;
        if (!source) {
          // QUAL is blank, but AA may still rank the model without an
          // effort label. Say so here: otherwise the blank cell reads as
          // missing data rather than a deliberate gate.
          var unlabeled = row.reasoning && row.reasoning.aa_unlabeled;
          var unlabeledIndex = finiteNumber(unlabeled && unlabeled.index);
          if (unlabeledIndex === null) return null;
          var checked = qualityAsOf("aa", unlabeled.as_of);
          return (
            "AA measures this model at " +
            pyFixed(unlabeledIndex, 1) +
            (checked === null ? "" : ", checked " + checked) +
            ", without a reasoning-effort label, so QUAL stays blank; " +
            "it may not be the quality you get at a selectable effort."
          );
        }
        var text = "AA intelligence index - source: " + String(source);
        var asOf = qualityAsOf(source, row.quality_as_of);
        if (asOf !== null) {
          text += (source === "aa" ? ", checked " : ", as of ") + asOf;
        }
        var reasoning = row.reasoning || {};
        var defaultEffort = row.default_effort || reasoning.default_effort;
        if (row.quality_effort) {
          text += ", measured at " + row.quality_effort;
          if (row.quality_effort === defaultEffort) {
            text += " (OpenRouter's effective default)";
          }
        }
        if (row.quality_drift === true && Number.isFinite(row.quality_or_index)) {
          var orDate = qualityAsOf("openrouter", row.quality_or_as_of);
          text +=
            " differs from OpenRouter snapshot (" +
            row.quality_or_index.toFixed(1) +
            (orDate === null ? "" : ", " + orDate) +
            ")";
        } else if (
          row.quality_drift !== true &&
          row.quality_drift !== false &&
          Number.isFinite(row.quality_or_index)
        ) {
          var refDate = qualityAsOf("openrouter", row.quality_or_as_of);
          text +=
            "; OpenRouter snapshot (" +
            row.quality_or_index.toFixed(1) +
            (refDate === null ? "" : ", " + refDate) +
            ") is not like-for-like: its reasoning effort is not reported";
        }
        if (
          source === "openrouter" &&
          row.quality_effort &&
          row.quality_effort !== "none" &&
          !Number.isFinite(row.quality_or_index)
        ) {
          text +=
            "; OpenRouter does not report the reasoning effort of its " +
            "snapshot: not like-for-like";
        }
        return text;
      }

      // endpointRows(catalogEntry): normalized rows for the per-model
      // endpoint disclosure. Nulls map to their display defaults (no
      // pricing -> "-", no status -> "non-live", no tier -> "-", no
      // quantization -> "unknown"); the baseline endpoint is the one whose
      // per-1M in/out prices match the entry's baseline pricing scalars
      // (the pricing block IS the baseline endpoint's pricing, so exactly
      // one endpoint should match -- if more do, the marker is withheld:
      // ambiguous beats wrong).
      function endpointRows(entry) {
        if (!entry || !Array.isArray(entry.endpoints)) return [];
        var base = entry.pricing || {};
        var baseIn =
          typeof base.input_per_1m === "number" && isFinite(base.input_per_1m)
            ? base.input_per_1m
            : null;
        var baseOut =
          typeof base.output_per_1m === "number" && isFinite(base.output_per_1m)
            ? base.output_per_1m
            : null;
        var matches = 0;
        var rows = entry.endpoints.map(function (endpoint) {
          var pricing = endpoint.pricing || {};
          var prompt =
            typeof pricing.prompt === "number" && isFinite(pricing.prompt)
              ? pricing.prompt
              : null;
          var completion =
            typeof pricing.completion === "number" && isFinite(pricing.completion)
              ? pricing.completion
              : null;
          var isBaseline = false;
          if (
            prompt !== null &&
            completion !== null &&
            baseIn !== null &&
            baseOut !== null &&
            Math.abs(prompt * 1e6 - baseIn) < 1e-9 &&
            Math.abs(completion * 1e6 - baseOut) < 1e-9
          ) {
            isBaseline = true;
            matches += 1;
          }
          return {
            provider: endpoint.provider_name || "unknown",
            badge: endpoint.quantization || "unknown",
            priceIn: prompt === null ? "-" : fmtPrice(prompt * 1e6),
            priceOut: completion === null ? "-" : fmtPrice(completion * 1e6),
            // status 0 is LIVE -- it is falsy, so `|| "non-live"` used to
            // mislabel every live endpoint. Null/undefined (not recorded)
            // renders "non-live"; anything else is the raw status code.
            status:
              endpoint.status === null || endpoint.status === undefined
                ? "non-live"
                : endpoint.status === 0
                  ? "live"
                  : endpoint.status,
            context: endpoint.context_length,
            tier: endpoint.service_tier || "-",
            isBaseline: isBaseline,
          };
        });
        if (matches > 1) {
          rows.forEach(function (r) {
            r.isBaseline = false;
          });
        }
        return rows;
      }

      // CTX cell caveat: shown only when the annotations flag says the
      // baseline endpoint's context is below the catalog context. Names the
      // baseline endpoint's context and the catalog context so the reader
      // knows the ranked price was quoted at the smaller window. No
      // baseline endpoint (or no context on it) -> no caveat.
      function fmtCtxCaveat(entry, row) {
        if (
          !entry ||
          !entry.annotations ||
          entry.annotations.baseline_context_below_catalog !== true
        ) {
          return null;
        }
        if (
          !row ||
          row.context_tokens === null ||
          row.context_tokens === undefined
        ) {
          return null;
        }
        var rows = endpointRows(entry);
        var baseline = null;
        for (var i = 0; i < rows.length; i++) {
          if (rows[i].isBaseline) {
            baseline = rows[i];
            break;
          }
        }
        if (
          !baseline ||
          baseline.context === null ||
          baseline.context === undefined
        ) {
          return null;
        }
        return (
          "Ranked price from a " +
          fmtContext(baseline.context) +
          "-token endpoint; model supports " +
          fmtContext(row.context_tokens) +
          "."
        );
      }

      function topModel() {
        var rows = data && data.priorities[current];
        if (!rows || !rows.length) return null;
        // provider-qualified id (opencode-ready); fall back for stale data
        return rows[0].opencode_model || rows[0].model;
      }

      function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          return navigator.clipboard.writeText(text).then(
            function () {
              return true;
            },
            function () {
              return legacyCopy(text);
            },
          );
        }
        return Promise.resolve(legacyCopy(text));
      }

      function legacyCopy(text) {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        var ok = false;
        try {
          ok = document.execCommand("copy");
        } catch (err) {
          ok = false;
        }
        ta.remove();
        return ok;
      }

      function flash(btn, ok) {
        var old = btn.textContent;
        btn.textContent = ok ? "Copied!" : "Copy failed";
        setTimeout(function () {
          btn.textContent = old;
        }, 1500);
      }

      function wireCopy(btn, getText) {
        btn.addEventListener("click", function () {
          var text = getText();
          if (text === null || text === undefined) return;
          copyText(text).then(function (ok) {
            flash(btn, ok);
          });
        });
      }

      var DEFAULT_MAIN_ROWS = 10;
      var DEFAULT_DEAL_ROWS = 5;
      var FULL_DEAL_ROWS = 10;
      // Reasoning-effort display order (deepest first), mirroring
      // model_compare.AA_EFFORT_ORDER. Kept in sync with the CLI.
      var EFFORT_ORDER = [
        "max",
        "xhigh",
        "high",
        "medium",
        "low",
        "minimal",
        "none",
        "reasoning",
      ];
      // Column counts of the main ranking table; the endpoint disclosure
      // row spans all visible ones. Kept in sync with the markup's <th>
      // count by web/test_site_js.py -- bump both when a column is added.
      var MAIN_TABLE_COLS = 14;
      // Simple view shows rank, 7-day, model, quality, in, out, discount,
      // age: the expert count minus the six hidden columns below.
      var SIMPLE_TABLE_COLS = 8;
      // Columns absent from the simple view, keyed by their cells[] index
      // (render() skips the cell; the markup's <th> and the legend hints
      // carry the matching hide-simple class). DEPTH, PREC, TIER, CTX,
      // SCORE, TASK$.
      var SIMPLE_HIDDEN_CELLS = {
        2: true,
        7: true,
        8: true,
        9: true,
        11: true,
        12: true,
      };
      var mainExpanded = false;
      var dealsExpanded = false;

      // Display-only transform: the page shows the typographic glyph; the
      // data strings stay ASCII (CLI parity).
      function displayTier(ascii) {
        return ascii.replace(">=", "≥");
      }

      function tierCellData(row) {
        var note = row.tier_note || row.sched_note || "--";
        var hovers = [];
        if (row.tier_note) {
          var endsK = row.tier_note.charAt(row.tier_note.length - 1) === "k";
          var size = parseInt(row.tier_note.slice(1), 10);
          var hover =
            "Charges a higher rate once the prompt exceeds " +
            size +
            (endsK ? "k" : "") +
            " tokens; price shown: that long-context rate.";
          // Below-threshold (base) prices ride along when the row carries
          // them: formatted by the same fmtPrice as the table columns.
          // Finite and non-negative: strings, NaN, +/-Infinity, negatives
          // and half-present pairs must drop the sentence, not leak
          // "NaN"/"Infinity" into the tooltip.
          if (
            Number.isFinite(row.base_input_usd_per_m) &&
            row.base_input_usd_per_m >= 0 &&
            Number.isFinite(row.base_output_usd_per_m) &&
            row.base_output_usd_per_m >= 0
          ) {
            hover +=
              " The short-context rate is $" +
              fmtPrice(row.base_input_usd_per_m) +
              " in / $" +
              fmtPrice(row.base_output_usd_per_m) +
              " out per 1M tokens.";
          }
          hovers.push(hover);
        }
        if (row.sched_note === "SCHED") {
          hovers.push(
            "Price varies on a schedule by less than 1%; price shown: the " +
              "standard rate.",
          );
        } else if (row.sched_note) {
          hovers.push(
            "Runs up to " +
              row.sched_note.slice(1) +
              " cheaper during scheduled off-peak hours" +
              (row.sched_detail ? " (" + row.sched_detail + ")" : "") +
              "; price shown: the standard rate.",
          );
        }
        return {
          text: displayTier(note),
          hover: hovers.length ? hovers.join(" ") : null,
        };
      }

      // Disclosure note: shown instead of silence when the entry's pricing
      // scalars fail to pin exactly one baseline endpoint -- ambiguous
      // (several match) or empty (none do); the "✓" would be a guess either
      // way. Silent when there is nothing to mark (no endpoints at all).
      function baselineNote(entry) {
        if (
          !entry ||
          !Array.isArray(entry.endpoints) ||
          !entry.endpoints.length
        ) {
          return null;
        }
        var matches = endpointRows(entry).filter(function (r) {
          return r.isBaseline;
        }).length;
        if (matches === 1) return null;
        return "baseline endpoint ambiguous — no marker";
      }

      function endpointRow(entry, catalogLoaded) {
        var tr = document.createElement("tr");
        tr.className = "endpoint-row";
        var td = document.createElement("td");
        td.colSpan = view === "simple" ? SIMPLE_TABLE_COLS : MAIN_TABLE_COLS;
        if (!catalogLoaded) {
          // Artifact-level: catalog.json itself is missing or failed to
          // load -- distinct from the model-level join miss below.
          td.textContent = "endpoint details unavailable — catalog fetch failed";
          tr.appendChild(td);
          return tr;
        }
        if (!entry) {
          td.textContent = "No endpoint data for this model.";
          tr.appendChild(td);
          return tr;
        }
        // Section heading, parallel to the Depth block's heading above.
        var heading = document.createElement("p");
        heading.className = "detail-heading";
        heading.textContent = "Providers";
        td.appendChild(heading);
        var table = document.createElement("table");
        var thead = document.createElement("thead");
        var headTr = document.createElement("tr");
        // Sentence-case headers (unabbreviated) with per-column tooltips,
        // the main tables' header pattern.
        [
          ["Provider", "Endpoint provider (gateway) serving this model."],
          ["Precision", "Quantization the endpoint's price is quoted at."],
          ["In $/M", "Input price at this endpoint, USD per 1M tokens."],
          ["Out $/M", "Output price at this endpoint, USD per 1M tokens."],
          ["Status", "Endpoint availability."],
          ["Context", "Context window at this endpoint, in tokens."],
          [
            "Tier",
            "How this endpoint's pricing varies: a token threshold means a " +
              "higher rate above it; a percentage means cheaper off-peak hours.",
          ],
          ["Baseline", "✓ marks the endpoint this model's ranked price comes from."],
        ].forEach(function (cell) {
          var th = document.createElement("th");
          th.textContent = cell[0];
          th.title = cell[1];
          headTr.appendChild(th);
        });
        thead.appendChild(headTr);
        table.appendChild(thead);
        var tbody = document.createElement("tbody");
        endpointRows(entry).forEach(function (er) {
          var row = document.createElement("tr");
          [
            er.provider,
            er.badge,
            er.priceIn,
            er.priceOut,
            er.status,
            er.context === null || er.context === undefined
              ? "-"
              : fmtContext(er.context),
            er.tier,
            er.isBaseline ? "✓" : "",
          ].forEach(function (cell) {
            var c = document.createElement("td");
            c.textContent = cell;
            row.appendChild(c);
          });
          tbody.appendChild(row);
        });
        table.appendChild(tbody);
        td.appendChild(table);
        var note = baselineNote(entry);
        if (note) {
          var p = document.createElement("p");
          p.className = "endpoint-note";
          p.textContent = note;
          td.appendChild(p);
        }
        tr.appendChild(td);
        return tr;
      }

      function depthSelectText(reasoning, depth) {
        if (depth.effort === "reasoning") return "-";
        var supported = Array.isArray(reasoning.supported_efforts)
          ? reasoning.supported_efforts
          : [];
        if (
          depth.effort === "none" &&
          !supported.length &&
          isNonReasoningBlock(reasoning)
        ) {
          return "-";
        }
        return "not selectable";
      }

      // depthTable(entry): the expanded row's Depth section table. Reads
      // the catalog entry's reasoning block (data.json rows stay lean and
      // drop it). Selectable efforts get a copy button for the OpenCode
      // variant id; an AA-measured effort OpenRouter cannot run stays for
      // transparency with a "not selectable" note and no button.
      function depthTable(entry) {
        var reasoning = entry && entry.reasoning;
        if (!reasoning) return null;
        var rows = depthRows({
          quality_effort: entry.quality_effort,
          default_effort: reasoning.default_effort,
          supported_efforts: reasoning.supported_efforts,
          reasoning: reasoning,
        });
        if (!rows.length) return null;
        var table = document.createElement("table");
        table.className = "depth-table";
        var thead = document.createElement("thead");
        var headTr = document.createElement("tr");
        // Sentence-case headers with the explanatory note moved into
        // per-column tooltips (the main tables' header pattern).
        [
          [
            "Effort",
            "Reasoning effort this row scores; * = OpenRouter's default. " +
              "OpenRouter's $/M prices are the same at every depth - only " +
              "AA's measurements differ.",
          ],
          ["AA index", "AA intelligence index measured at this effort; - = not measured."],
          [
            "AA $ per task",
            "AA's measured benchmark cost at this effort (AA's reference " +
              "price, not your session cost).",
          ],
          ["Vs default", "This depth's AA $ per task divided by the default depth's."],
          [
            "Select",
            "Copy the OpenCode model id that pins this depth (#effort). " +
              "'not selectable' = AA measured this effort but OpenRouter " +
              "cannot run it.",
          ],
        ].forEach(function (cell) {
          var th = document.createElement("th");
          th.textContent = cell[0];
          th.title = cell[1];
          headTr.appendChild(th);
        });
        thead.appendChild(headTr);
        table.appendChild(thead);
        var tbody = document.createElement("tbody");
        rows.forEach(function (depth) {
          var tr = document.createElement("tr");
          if (depth.selected) tr.className = "depth-selected";
          [
            depth.effort + (depth.default ? "*" : ""),
            depth.index === null ? "-" : pyFixed(depth.index, 1),
            depth.cost === null ? "-" : pyFixed(depth.cost, 2),
            fmtDepthRatio(depth.ratio),
          ].forEach(function (cell) {
            var td = document.createElement("td");
            td.textContent = cell;
            tr.appendChild(td);
          });
          var select = document.createElement("td");
          if (depth.selectable) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "copy-depth";
            btn.textContent = "#" + depth.effort;
            wireCopy(btn, function () {
              return "openrouter/" + entry.id + "#" + depth.effort;
            });
            select.appendChild(btn);
          } else {
            select.className = "not-selectable";
            select.textContent = depthSelectText(reasoning, depth);
          }
          tr.appendChild(select);
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        return table;
      }

      // The depth disclosure row. Kept separate from endpointRow so each
      // section degrades on its own (catalog failure vs a model-level join
      // miss); render() appends this one first, then the endpoint table.
      function depthDetailRow(entry, catalogLoaded) {
        var tr = document.createElement("tr");
        tr.className = "endpoint-row depth-row";
        var td = document.createElement("td");
        td.colSpan = view === "simple" ? SIMPLE_TABLE_COLS : MAIN_TABLE_COLS;
        if (!catalogLoaded) {
          td.textContent = "depth details unavailable - catalog fetch failed";
          tr.appendChild(td);
          return tr;
        }
        if (!entry || !entry.reasoning) {
          td.textContent = "no depth data for this model.";
          tr.appendChild(td);
          return tr;
        }
        var heading = document.createElement("p");
        heading.className = "detail-heading";
        heading.textContent = "Depth";
        td.appendChild(heading);
        var note = unlabeledNote(entry.reasoning.aa_unlabeled);
        if (note !== null) {
          var noteP = document.createElement("p");
          noteP.className = "endpoint-note";
          noteP.textContent = note;
          td.appendChild(noteP);
        }
        var table = depthTable(entry);
        if (!table) {
          // A reference note already says AA measured this model; the
          // blanket "no depth data" line would contradict it.
          if (note === null) {
            var empty = document.createElement("p");
            empty.className = "endpoint-note";
            empty.textContent = "no depth data for this model.";
            td.appendChild(empty);
          }
        } else {
          td.appendChild(table);
        }
        tr.appendChild(td);
        return tr;
      }

      function render() {
        var rows = (data && data.priorities[current]) || [];
        var shown = mainExpanded ? rows : rows.slice(0, DEFAULT_MAIN_ROWS);
        var expandMain = document.getElementById("expand-main");
        expandMain.hidden = rows.length <= DEFAULT_MAIN_ROWS;
        expandMain.textContent = mainExpanded ? "Show fewer" : "Show more";
        els.tbody.textContent = "";
        var baseline = null;
        if (historyData && historyData.snapshots) {
          var today = data
            ? data.generated_at.slice(0, 10)
            : new Date().toISOString().slice(0, 10);
          var d = new Date(today + "T00:00:00Z");
          d.setUTCDate(d.getUTCDate() - 7);
          baseline =
            historyData.snapshots[d.toISOString().slice(0, 10)] || null;
        }
        shown.forEach(function (row, i) {
          var tr = document.createElement("tr");
          var movement = { cls: "", text: "" };
          if (baseline) {
            var prevRank = null;
            ((baseline.tabs || {})[current] || []).forEach(function (r) {
              if (r.id === row.model) prevRank = r.rank;
            });
            if (prevRank === null) movement = { cls: "new", text: "new" };
            else if (prevRank > i + 1)
              movement = { cls: "up", text: "▴" + (prevRank - i - 1) };
            else if (prevRank < i + 1)
              movement = { cls: "down", text: "▾" + (i + 1 - prevRank) };
            else movement = { cls: "flat", text: "•" };
          }
          var tier = tierCellData(row);
          var entry = null;
          if (catalogData && Array.isArray(catalogData.models)) {
            for (var j = 0; j < catalogData.models.length; j++) {
              if (catalogData.models[j].id === row.model) {
                entry = catalogData.models[j];
                break;
              }
            }
          }
          // data.json drops the nested reasoning block (it stays lean); the
          // depth table and the DEPTH/QUAL hints join it back from the
          // catalog entry looked up above.
          var depthRow = Object.assign(
            {},
            row,
            entry && entry.reasoning ? { reasoning: entry.reasoning } : {},
          );
          var precTooltip = fmtQuantTooltip(row);
          var ctxCaveat = fmtCtxCaveat(entry, row);
          var qualityTip = qualityTooltip(depthRow);
          var depthTip = depthTooltip(depthRow);
          var qualityScore =
            row.quality_index === null ? "-" : row.quality_index.toFixed(1);
          var simpleQuality = qualityScore;
          if (row.quality_index !== null && depthRow.quality_effort) {
            simpleQuality += " · " + depthRow.quality_effort;
          }
          var cells = [
            String(i + 1),
            row.model,
            fmtEffort(depthRow),
            view === "simple" ? simpleQuality : qualityScore,
            fmtPrice(row.input_usd_per_m),
            fmtPrice(row.output_usd_per_m),
            row.discount_pct || "--",
            fmtPrec(row),
            tier.text,
            fmtContext(row.context_tokens),
            fmtAge(row.age_days),
            row.score.toFixed(3),
            fmtTaskCost(row.session_cost_usd),
          ];
          cells.forEach(function (cell, idx) {
            // Simple view drops the six expert-only columns; the <th>
            // markup and the legend hints carry the matching hide-simple
            // class, so the row and its headers stay aligned.
            if (view === "simple" && SIMPLE_HIDDEN_CELLS[idx]) return;
            if (idx === 1) {
              var moveTd = document.createElement("td");
              if (movement.cls) moveTd.className = "movement " + movement.cls;
              moveTd.textContent = movement.text;
              tr.appendChild(moveTd);
            }
            var td = document.createElement("td");
            td.textContent = cell;
            if (idx === 2) {
              if (depthTip) td.title = depthTip;
            } else if (idx === 3) {
              if (qualityTip) td.title = qualityTip;
            } else if (idx === 7) {
              if (precTooltip) td.title = precTooltip;
            } else if (idx === 8) {
              td.className = "tier";
              if (tier.hover) td.title = tier.hover;
            } else if (idx === 9) {
              if (ctxCaveat) {
                td.title = ctxCaveat;
                td.className = "caveat";
              }
            } else if (idx === 12) {
              if (cell === "—") {
                td.title = "Not available this refresh.";
                td.className = "na";
              }
            }
            tr.appendChild(td);
          });
          // Keyboard-operable disclosure toggle: the row is a button, so
          // Enter and Space both flip it (Space must not scroll the page).
          tr.tabIndex = 0;
          tr.setAttribute("role", "button");
          tr.setAttribute(
            "aria-expanded",
            endpointExpanded[row.model] ? "true" : "false"
          );
          tr.addEventListener("click", function () {
            endpointExpanded[row.model] = !endpointExpanded[row.model];
            render();
          });
          tr.addEventListener("keydown", function (ev) {
            if (ev.key !== "Enter" && ev.key !== " ") return;
            if (ev.key === " ") ev.preventDefault();
            endpointExpanded[row.model] = !endpointExpanded[row.model];
            render();
          });
          els.tbody.appendChild(tr);
          if (endpointExpanded[row.model]) {
            els.tbody.appendChild(depthDetailRow(entry, catalogData));
            els.tbody.appendChild(endpointRow(entry, catalogData));
          }
        });
        var model = topModel();
        var hasTop = model !== null;
        els.copyModel.disabled = !hasTop;
        els.copyCommand.disabled = !hasTop;
        els.modelValue.textContent = model === null ? "—" : model;
        els.commandValue.textContent =
          model === null ? "—" : OPENCODE_PREFIX + model;
        renderDeals();
      }

      function renderDeals() {
        // render() also runs when history.json lands first: with data.json
        // not yet loaded (or failed) the section keeps its initial hidden
        // state instead of claiming there are no deals.
        if (!data) return;
        var deals = data.schedules || [];
        var empty = document.getElementById("deals-empty");
        var intro = document.getElementById("deals-intro");
        var table = document.getElementById("deals");
        var button = document.getElementById("expand-deals");
        var tbody = document.getElementById("deals-tbody");
        if (!deals.length) {
          intro.hidden = true;
          table.hidden = true;
          button.hidden = true;
          empty.hidden = false;
          return;
        }
        empty.hidden = true;
        intro.hidden = false;
        table.hidden = false;
        var shown = dealsExpanded
          ? deals.slice(0, FULL_DEAL_ROWS)
          : deals.slice(0, DEFAULT_DEAL_ROWS);
        tbody.textContent = "";
        shown.forEach(function (deal) {
          var tr = document.createElement("tr");
          // Null-guarded: the validator requires sched_note, but a
          // hand-edited catalog must not throw into the fetch .catch.
          var note = deal.sched_note ? String(deal.sched_note) : "";
          var qualityTip = qualityTooltip(deal);
          var qualityText =
            deal.quality === null || deal.quality === undefined
              ? "-"
              : deal.quality.toFixed(1);
          // Same effort tag as the main table's simple QUAL cell: the
          // schedule entry carries the depth its recorded quality used.
          if (
            deal.quality !== null &&
            deal.quality !== undefined &&
            deal.quality_effort
          ) {
            qualityText += " · " + deal.quality_effort;
          }
          var cells = [
            deal.model,
            note || "--",
            deal.sched_detail || "",
            qualityText,
            fmtPrice(deal.input_per_1m),
            fmtPrice(deal.output_per_1m),
            deal.context === null || deal.context === undefined
              ? "-"
              : fmtContext(deal.context),
            deal.score === null || deal.score === undefined
              ? "-"
              : deal.score.toFixed(3),
          ];
          cells.forEach(function (cell, idx) {
            var td = document.createElement("td");
            td.textContent = cell;
            if (idx === 1) {
              td.className = "offpeak";
              if (note) {
                var pct = note === "SCHED" ? "less than 1%" : note.slice(1);
                td.title =
                  "Cheapest scheduled rates: up to " +
                  pct +
                  " below the standard price during the listed windows.";
              }
            }
            if (idx === 2) {
              td.className = "schedule-cell";
              td.dataset.full = deal.sched_detail || "";
              var span = document.createElement("span");
              span.className = "schedule-text";
              span.textContent = deal.sched_detail || "";
              td.textContent = "";
              td.appendChild(span);
            }
            if (idx === 3) {
              if (qualityTip) td.title = qualityTip;
            }
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        button.hidden = deals.length <= DEFAULT_DEAL_ROWS;
        button.textContent = dealsExpanded ? "Show fewer" : "Show more";
        // Tooltip only when actually clipped (the inner span is what
        // clips); length fallback when the layout metrics are unavailable
        // (spec section 5.5).
        Array.prototype.forEach.call(
          tbody.querySelectorAll("td.schedule-cell"),
          function (cell) {
            var span = cell.querySelector(".schedule-text");
            if (!span) return;
            var clipped = span.scrollWidth > span.clientWidth;
            if (
              !clipped &&
              span.clientWidth === 0 &&
              cell.dataset.full.length > 25
            ) {
              clipped = true;
            }
            if (clipped) cell.title = cell.dataset.full;
          },
        );
      }

      function renderHighlights(sections) {
        var keys = ["week", "intelligence", "prices"];
        var complete = keys.every(function (key) {
          return typeof sections[key] === "string" && sections[key].length > 0;
        });
        if (!complete) return;
        document.getElementById("highlights").hidden = false;
        keys.forEach(function (key) {
          var el = document.getElementById("hl-" + key);
          if (!el) return;
          var safe = sections[key]
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
          el.innerHTML = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
        });
      }

      // Both artifact failures compose into one status line: the catalog
      // catch must not clobber a data.json error (the fetches are parallel,
      // so either may reject first).
      function renderStatus() {
        var parts = [];
        if (dataError !== null) {
          parts.push("Could not load data.json: " + dataError);
        }
        if (catalogError) {
          parts.push("endpoint details unavailable (catalog fetch failed)");
        }
        if (!parts.length) return;
        els.status.textContent = parts.join(" ; ");
        els.status.hidden = false;
      }

      // Footer stamps: the data.json generation time and the source
      // revision link. Extracted from the fetch chain so the flow is
      // unit-testable (web/test_site_js.py drives it under node).
      function renderFooter(data) {
        var stamp = new Date(data.generated_at).toLocaleString();
        els.generated.textContent = stamp;
        els.generatedTop.textContent = "last refresh: " + stamp;
        if (data.source_revision) {
          els.revBlock.hidden = false;
          els.revLink.href =
            "https://github.com/canonical/model-compare/commit/" +
            data.source_revision;
          els.revLink.textContent = data.source_revision;
        }
      }

      function init() {
        els.aliasHint.textContent = "alias oc-best='" + CURL_COMMAND + "'";
        els.curlValue.textContent = CURL_COMMAND;
        els.copyCurl.disabled = false;
        wireCopy(els.copyModel, topModel);
        wireCopy(els.copyCommand, function () {
          var model = topModel();
          return model === null ? null : OPENCODE_PREFIX + model;
        });
        wireCopy(els.copyCurl, function () {
          return CURL_COMMAND;
        });

        Array.prototype.forEach.call(
          document.querySelectorAll("#tabs button[data-priority]"),
          function (btn) {
            btn.addEventListener("click", function () {
              current = btn.dataset.priority;
              Array.prototype.forEach.call(
                document.querySelectorAll("#tabs button[data-priority]"),
                function (b) {
                  b.classList.toggle("active", b === btn);
                },
              );
              if (data) render();
            });
          },
        );

        // Restore the persisted view (consent-first cookie) before any
        // fetch: a "simple" cookie restores simple view, absence (or
        // anything else) restores the expert default.
        if (readCookie(COOKIE_NAME) === "simple") view = "simple";
        applyView();

        // Focus moves into the dialog when it opens (it carries
        // tabindex="-1") and back to the toggle that opened it when it
        // closes, so keyboard and screen-reader users see the
        // conversation. Escape dismisses without storing anything.
        function openConsentDialog() {
          var dlg = document.getElementById("cookie-dialog");
          dlg.hidden = false;
          dlg.focus();
        }
        function closeConsentDialog() {
          var dlg = document.getElementById("cookie-dialog");
          if (dlg.hidden) return;
          dlg.hidden = true;
          document.getElementById("view-simple").focus();
        }
        document
          .getElementById("view-simple")
          .addEventListener("click", function () {
            setView("simple");
            if (shouldAskForConsent()) {
              openConsentDialog();
            }
          });
        document
          .getElementById("view-expert")
          .addEventListener("click", function () {
            setView("expert");
            closeConsentDialog();
            if (readCookie(COOKIE_NAME) === "simple") clearViewCookie();
          });
        document
          .getElementById("cookie-ok")
          .addEventListener("click", function () {
            writeViewCookie();
            consentRejected = false;
            closeConsentDialog();
          });
        document
          .getElementById("cookie-reject")
          .addEventListener("click", function () {
            consentRejected = true;
            closeConsentDialog();
          });
        document
          .getElementById("cookie-dialog")
          .addEventListener("keydown", function (ev) {
            if (ev.key !== "Escape") return;
            // A dismissal is not a consent: same contract as Reject.
            consentRejected = true;
            closeConsentDialog();
          });

        document
          .getElementById("expand-main")
          .addEventListener("click", function () {
            mainExpanded = !mainExpanded;
            if (data) render();
          });
        document
          .getElementById("expand-deals")
          .addEventListener("click", function () {
            dealsExpanded = !dealsExpanded;
            renderDeals();
          });

        fetch("data.json", { cache: "no-cache" })
          .then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          })
          .then(function (payload) {
            data = payload;
            renderFooter(data);
            render();
          })
          .catch(function (err) {
            dataError = err.message;
            renderStatus();
          });

        fetch("history.json", { cache: "no-cache" })
          .then(function (r) {
            return r.ok ? r.json() : null;
          })
          .then(function (payload) {
            historyData = payload && payload.snapshots ? payload : null;
            render();
          })
          .catch(function () {
            historyData = null;
          });

        fetch("highlights.json", { cache: "no-cache" })
          .then(function (r) {
            return r.ok ? r.json() : null;
          })
          .then(function (payload) {
            if (payload && payload.sections) renderHighlights(payload.sections);
          })
          .catch(function () {});

        // catalog.json is a separate artifact: a failure must never block
        // the data.json table (degraded mode), just surface a one-line
        // notice and leave the per-model disclosure empty.
        fetch("catalog.json", { cache: "no-cache" })
          .then(function (r) {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          })
          .then(function (payload) {
            catalogData = payload;
            render();
          })
          .catch(function () {
            catalogError = true;
            renderStatus();
          });
      }

      init();
