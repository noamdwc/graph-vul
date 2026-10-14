// One JVM per worker; each source is still parsed into a separate CPG.
import io.joern.c2cpg.{C2Cpg, Config}
import io.joern.x2cpg.X2Cpg
import io.shiftleft.semanticcpg.language.*
import java.nio.file.{Files, Path, StandardOpenOption}
import scala.jdk.CollectionConverters.*

@main def exec(listFile: String, outputDir: String) = {
  val frontend = new C2Cpg()
  try {
    Files.readAllLines(Path.of(listFile)).asScala.foreach { filename =>
      val source = Path.of(filename)
      val id = source.getFileName.toString.stripSuffix(".c")
      val out = Path.of(outputDir, id)
      Files.createDirectory(out)
      val started = System.nanoTime()
      val graph = frontend.createCpg(Config().withInputPath(filename).withOutputPath(out.resolve("cpg.bin").toString)).get
      val counts = try {
        X2Cpg.applyDefaultOverlays(graph)
        val allMethods = graph.method.toList
        allMethods.zipWithIndex.foreach { case (method, index) =>
          Files.writeString(out.resolve(s"$index-cpg.dot"), Iterator.single(method).dotCpg14.next(), StandardOpenOption.CREATE_NEW)
        }
        (allMethods.size, allMethods.count(m => !m.isExternal && m.name != "<global>"))
      } finally graph.close()
      // Mark completion only after every DOT export and CPG close succeeds.
      val status = ujson.Obj("sample_id" -> id, "methods" -> counts._1,
                            "defined_functions" -> counts._2,
                            "seconds" -> ((System.nanoTime() - started) / 1e9))
      Files.writeString(out.resolve("status.json"), ujson.write(status), StandardOpenOption.CREATE_NEW)
      println(s"EXPORTED $id ${status("seconds")}")
    }
  } finally frontend.close()
}
