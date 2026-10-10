// js_stubs.js — minimal browser shims for evaluating the inline site
// script under node (web/test_site_js.py prepends this file to the
// <script> block extracted from web/site/index.html).
//
// Only what the script touches at evaluation time: element lookups,
// tab/expander wiring, and the three fetch() calls (rejected, so the
// page ends in its error path and pure helpers stay callable).

function stubElement() {
  return {
    textContent: "",
    hidden: false,
    disabled: false,
    title: "",
    className: "",
    dataset: {},
    style: {},
    classList: {
      toggle: function () {},
      add: function () {},
      remove: function () {},
    },
    addEventListener: function () {},
    appendChild: function () {},
    remove: function () {},
    select: function () {},
  };
}

var window = { location: { href: "http://127.0.0.1:8734/index.html" } };
var document = {
  getElementById: function () {
    return stubElement();
  },
  createElement: function () {
    return stubElement();
  },
  querySelectorAll: function () {
    return [];
  },
  querySelector: function () {
    return null;
  },
  body: stubElement(),
};
var navigator = {};
var fetch = function () {
  return Promise.reject(new Error("stub: fetch disabled in JS tests"));
};
