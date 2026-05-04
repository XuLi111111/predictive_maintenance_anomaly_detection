import Footer from "../components/Footer";
import Nav from "../components/Nav";

export default function Upload() {
  return (
    <>
      <Nav active="upload" />
      <section className="stub">
        Upload page — to be implemented by teammate.
        <br />
        Required: drop zone, schema validation, model picker, perf panel, run
        button (FR-01–11, FR-21).
      </section>
      <Footer />
    </>
  );
}
